"""Config-driven regressor experiments for QualiSR-Lab."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import warnings
from copy import deepcopy
from fnmatch import fnmatch
from functools import reduce
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR, SVR

from qualisr_lab.profiling import (
    build_regressor_profile_row,
    build_regressor_total_profile,
    is_regressor_profiling_enabled,
    load_feature_profile_summary,
    resolve_regressor_profile_path,
    resolve_regressor_total_profile_path,
)


PRETTY_FEATURE_NAMES = {
    "catboost": "CatBoost",
    "randomforest": "Random Forest",
    "xgb": "XGBoost",
    "gradientboosting": "Gradient Boosting",
    "gradientboost": "Gradient Boosting",
    "lgbm": "LightGBM",
    "lightgbm": "LightGBM",
    "linear": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "svr": "SVR",
    "linear_svr": "Linear SVR",
    "mlp": "MLP",
    "best": "Best",
    "mean_features": "Mean Features",
    "median_features": "Median Features",
    "vgg": "VGG",
    "resnet": "ResNet",
    "siglip": "SigLIP",
    "nr": "NR",
    "fr": "FR",
    "musiq": "MUSIQ",
    "unique": "UNIQUE",
    "arniqa": "ARNIQA",
    "qalign": "Q-Align",
    "paq2piq": "PaQ-2-PiQ",
    "stlpips-vgg": "STLPIPS-VGG",
    "lpips-vgg": "LPIPS-VGG",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "pieapp": "PieAPP",
    "ahiq": "AHIQ",
    "rlfn": "RLFN",
    "span": "SPAN",
    "bicubic": "Bicubic",
    "gt": "GT",
    "content_fidelity": "Content Fidelity",
    "perceptual_enhancement": "Perceptual Enhancement",
    "final_rr_score": "Final RR Score",
    "min": "Min",
    "max": "Max",
    "mean": "Mean",
    "median": "Median",
    "std": "Std",
    "p05": "P05",
    "p95": "P95",
    "area00": "Area 0.00",
    "area05": "Area 0.05",
    "area075": "Area 0.75",
}

DEFAULT_NR_METRICS = ("musiq", "arniqa", "qalign", "unique", "paq2piq")
DEFAULT_FR_METRICS = ("psnr", "ssim", "lpips-vgg", "stlpips-vgg", "pieapp", "ahiq")

PCA_FEATURE_RE = re.compile(r"^(?P<family>vgg|resnet)(?:_pca)?[_-]?(?P<component>\d+)$", re.IGNORECASE)


def configured_pretty_names(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    names = {str(key): str(value) for key, value in PRETTY_FEATURE_NAMES.items()}
    if not cfg:
        return names

    pretty_cfg = cfg.get("pretty_names", {})
    if not isinstance(pretty_cfg, dict):
        return names

    for section in ["features", "models", "metrics", "references", "names"]:
        section_names = pretty_cfg.get(section, {})
        if isinstance(section_names, dict):
            names.update({str(key): str(value) for key, value in section_names.items()})

    flat_names = {
        str(key): str(value)
        for key, value in pretty_cfg.items()
        if isinstance(value, str)
    }
    names.update(flat_names)
    return names


def lookup_pretty_name(name: str, cfg: dict[str, Any] | None = None) -> str | None:
    names = configured_pretty_names(cfg)
    if name in names:
        return names[name]

    lower_names = {key.lower(): value for key, value in names.items()}
    return lower_names.get(name.lower())


def get_pretty_feature(name: Any, cfg: dict[str, Any] | None = None) -> str:
    raw = str(name)
    configured_name = lookup_pretty_name(raw, cfg)
    if configured_name is not None:
        return configured_name

    lower = raw.lower()
    for ref in ["bicubic", "rlfn", "span", "gt"]:
        suffix = f"_{ref}"
        if lower.endswith(suffix):
            prefix = raw[: -len(suffix)]
            prefix_pretty = lookup_pretty_name(prefix, cfg) or prefix
            ref_pretty = lookup_pretty_name(ref, cfg) or ref.upper()
            return f"{prefix_pretty} + {ref_pretty}"

    match = PCA_FEATURE_RE.match(lower)
    if match:
        family = match.group("family")
        family_pretty = lookup_pretty_name(family, cfg) or {"vgg": "VGG", "resnet": "ResNet"}[family]
        component = int(match.group("component"))
        return f"{family_pretty} PC{component}"

    return raw


def get_pretty_labels(names: Any, cfg: dict[str, Any] | None = None) -> list[str]:
    return [get_pretty_feature(name, cfg) for name in names]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def safe_corr(y_true: Any, y_pred: Any) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true_arr.size < 2 or y_pred_arr.size < 2:
        return np.nan, np.nan
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            "safe_corr received arrays with different lengths: "
            f"y_true={y_true_arr.size}, y_pred={y_pred_arr.size}"
        )

    if np.allclose(y_true_arr, y_true_arr[0]) or np.allclose(y_pred_arr, y_pred_arr[0]):
        return np.nan, np.nan

    return (
        float(pearsonr(y_pred_arr, y_true_arr).statistic),
        float(spearmanr(y_pred_arr, y_true_arr).statistic),
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def regressor_output_dirs(out_dir: Path) -> dict[str, Path]:
    """Create and return the organized output folders for one regressor run."""
    names = {
        "metadata": "metadata",
        "predictions": "predictions",
        "correlations": "correlations",
        "importances": "importances",
        "shap": "shap",
        "feature_analysis": "feature_analysis",
        "profiling": "profiling",
        "outliers": "outliers",
        "feature_selection": "feature_selection",
    }
    return {key: ensure_dir(out_dir / folder) for key, folder in names.items()}


def relativize_path_value(value: object) -> object:
    if pd.isna(value):
        return value

    raw = str(value).strip()
    if not raw:
        return value

    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw

    try:
        return Path(os.path.relpath(path, start=Path.cwd())).as_posix()
    except ValueError:
        return path.as_posix()


def relativize_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    path_columns = [
        column
        for column in result.columns
        if str(column) == "path" or str(column).endswith("_path") or str(column).endswith("_paths")
    ]
    for column in path_columns:
        result[column] = result[column].map(relativize_path_value)
    return result


def prepare_scores_file(cfg: dict[str, Any]) -> pd.DataFrame:
    prep_cfg = cfg["score_preparation"]
    raw_scores = pd.read_csv(cfg["paths"]["raw_scores"])

    method_col = prep_cfg["method_column"]
    case_col = prep_cfg["case_column"]
    score_col = prep_cfg["score_column"]

    method_map = {str(k).lower(): v for k, v in prep_cfg["method_map"].items()}
    mapped_methods = raw_scores[method_col].astype(str).str.lower().map(method_map)

    if mapped_methods.isna().any():
        missing = sorted(raw_scores.loc[mapped_methods.isna(), method_col].astype(str).unique().tolist())
        raise ValueError(f"Missing method_map entries for: {missing}")

    suffix = prep_cfg["name_suffix"]
    names = mapped_methods.astype(str) + "/" + raw_scores[case_col].astype(str) + suffix
    prepared = pd.DataFrame(
        {
            "name": names,
            cfg["dataset"]["score_column"]: raw_scores[score_col],
        }
    )

    score_path = Path(cfg["paths"]["scores"])
    ensure_dir(score_path.parent)
    prepared.to_csv(score_path, index=False)
    return prepared


def load_scores(cfg: dict[str, Any]) -> pd.DataFrame:
    if cfg["score_preparation"]["enabled"]:
        scores = prepare_scores_file(cfg)
    else:
        scores = pd.read_csv(cfg["paths"]["scores"])

    name_col = cfg["dataset"]["name_column"]
    score_col = cfg["dataset"]["score_column"]

    if name_col not in scores.columns:
        raise ValueError(f"Scores file must contain '{name_col}' column")

    if score_col not in scores.columns:
        fallback = [c for c in ["score", "scores", "mos", "mos_norm"] if c in scores.columns]
        if not fallback:
            raise ValueError(
                f"Scores file must contain '{score_col}' column. Available: {scores.columns.tolist()}"
            )
        scores = scores.rename(columns={fallback[0]: score_col})

    return scores[[name_col, score_col]].copy()


def build_sample_name(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    method_col = cfg["dataset"]["sr_method_column"]
    filename_col = cfg["dataset"]["sr_filename_column"]
    suffix = cfg["dataset"]["filename_suffix"]

    if method_col not in df.columns or filename_col not in df.columns:
        raise ValueError(
            f"Feature file must have '{method_col}' and '{filename_col}' columns. "
            f"Got: {df.columns.tolist()}"
        )

    stem = df[filename_col].astype(str).str.rsplit(".", n=1).str[0]
    return df[method_col].astype(str) + "/" + stem + suffix


def resolve_feature_path(feat_name: str, cfg: dict[str, Any]) -> Path:
    templates = cfg["features"]["feature_files"]
    if feat_name not in templates:
        raise KeyError(f"No path template configured for feature '{feat_name}'")

    try:
        pca_n = cfg["features"]["pca_n"]
    except KeyError:
        pca_n = 0

    return Path(
        templates[feat_name].format(
            features_root=cfg["paths"]["features_root"],
            pca_n=pca_n,
        )
    )


def resolve_configured_path(path_template: str, cfg: dict[str, Any]) -> Path:
    return Path(
        path_template.format(
            features_root=cfg["paths"]["features_root"],
            pca_n=cfg["features"]["pca_n"],
        )
    )


def keep_requested_fr_columns(df: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    refs_lower = [r.lower() for r in refs]
    keep_cols = ["name"]
    for col in df.columns:
        if col == "name":
            continue
        if any(col.lower().endswith("_" + ref) for ref in refs_lower):
            keep_cols.append(col)
    return df[keep_cols]


def load_feature_block(feat_name: str, cfg: dict[str, Any], valid_names: set[str]) -> pd.DataFrame:
    path = resolve_feature_path(feat_name, cfg)
    if not path.exists():
        raise FileNotFoundError(f"Feature file for '{feat_name}' not found: {path}")

    df = pd.read_csv(path)
    df["name"] = build_sample_name(df, cfg)

    drop_candidates = cfg["dataset"]["metadata_drop"]
    drop_existing = [c for c in drop_candidates if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    if feat_name == "fr":
        df = keep_requested_fr_columns(df, cfg["features"]["fr_refs"])

    return df[df["name"].isin(valid_names)].copy()


def load_stats_block(cfg: dict[str, Any], valid_names: set[str]) -> pd.DataFrame:
    stats_path = resolve_feature_path("stats", cfg)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    stats = pd.read_csv(stats_path)
    requested = ["name"] + cfg["features"]["stats_columns"]
    missing = [c for c in requested if c not in stats.columns]
    if missing:
        raise ValueError(f"Stats file is missing requested columns: {missing}")

    stats = stats[requested]
    return stats[stats["name"].isin(valid_names)].copy()


def build_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    scores = load_scores(cfg)
    valid_names = set(scores[cfg["dataset"]["name_column"]].tolist())

    frames = [scores]
    if cfg["features"]["include_stats"]:
        frames.append(load_stats_block(cfg, valid_names))

    for feat_name in cfg["features"]["include"]:
        frames.append(load_feature_block(feat_name, cfg, valid_names))

    dataset = reduce(lambda left, right: pd.merge(left, right, on="name", how="inner"), frames)

    try:
        existing_excludes = [c for c in cfg["features"]["exclude_columns"] if c in dataset.columns]
    except KeyError:
        existing_excludes = []
    if existing_excludes:
        dataset = dataset.drop(columns=existing_excludes)

    return dataset.sort_values("name").reset_index(drop=True)


def metric_comparison_column(item: dict[str, Any]) -> str:
    if "column" in item:
        return str(item["column"])

    metric = item.get("metric")
    if not metric:
        raise ValueError(f"Correlation metric item must define 'column' or 'metric': {item}")

    reference = item.get("reference")
    if reference:
        return f"{metric}_{reference}"
    return str(metric)


def metric_comparison_label(item: dict[str, Any], column: str, cfg: dict[str, Any]) -> str:
    if "label" in item:
        return str(item["label"])

    feature = get_pretty_feature(item.get("feature", ""), cfg).upper()
    reference = item.get("reference")
    metric = get_pretty_feature(item.get("metric", column), cfg)
    if reference:
        metric = f"{metric}+{get_pretty_feature(reference, cfg)}"
    if feature:
        return f"{metric} ({feature})"
    return metric


def load_metric_comparison_values(
    item: dict[str, Any],
    cfg: dict[str, Any],
    target_names: pd.Series,
) -> tuple[str, str, pd.Series]:
    if "path" in item:
        path = resolve_configured_path(item["path"], cfg)
    else:
        feature_name = item.get("feature")
        if feature_name is None:
            raise ValueError(f"Correlation metric item must define 'feature' or 'path': {item}")
        path = resolve_feature_path(str(feature_name), cfg)

    if not path.exists():
        raise FileNotFoundError(f"Correlation metric feature file not found: {path}")

    column = metric_comparison_column(item)
    values = pd.read_csv(path)
    values["name"] = build_sample_name(values, cfg)
    if column not in values.columns:
        raise ValueError(
            f"Correlation metric column '{column}' not found in {path}. "
            f"Available columns: {values.columns.tolist()}"
        )

    subset = values[["name", column]].copy()
    if subset["name"].duplicated().any():
        duplicates = sorted(subset.loc[subset["name"].duplicated(), "name"].unique().tolist())
        raise ValueError(f"Correlation metric file {path} has duplicate sample names: {duplicates[:10]}")

    aligned = target_names.to_frame(name="name").merge(subset, on="name", how="left")[column]
    label = metric_comparison_label(item, column, cfg)
    return label, column, aligned


def normalize_metric_values(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return numeric

    value_min = finite.min()
    value_max = finite.max()
    if np.isclose(value_min, value_max):
        normalized = pd.Series(np.nan, index=numeric.index, dtype=float)
    else:
        normalized = (numeric - value_min) / (value_max - value_min)

    if not higher_is_better:
        normalized = 1 - normalized

    return normalized


def compute_metric_comparisons(
    cfg: dict[str, Any],
    dataset: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict[str, Any]]:
    comparison_cfg = cfg.get("correlation_metrics", {})
    if not comparison_cfg.get("enabled", False):
        return []

    name_col = cfg["dataset"]["name_column"]
    target_names = dataset.loc[y_test.index, name_col].reset_index(drop=True)
    target_scores = y_test.reset_index(drop=True)

    rows = []
    for item in comparison_cfg.get("items", []):
        label, column, values = load_metric_comparison_values(item, cfg, target_names)
        higher_is_better = item.get("higher_is_better", True)
        normalized_values = normalize_metric_values(values, higher_is_better=higher_is_better)

        valid = target_scores.notna() & normalized_values.notna()
        if not valid.any():
            raise ValueError(f"Correlation metric '{label}' has no values aligned with the test split")

        plcc, srcc = safe_corr(target_scores[valid], normalized_values[valid])
        rows.append(
            {
                "model": label,
                "plcc": plcc,
                "srcc": srcc,
                "source": "metric",
                "feature": item.get("feature"),
                "column": column,
                "higher_is_better": higher_is_better,
                "normalized": True,
            }
        )

    return rows


def build_group_keys(names: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    segment_idx = cfg["dataset"]["group_segment_index"]
    remove_suffix = cfg["dataset"].get("group_remove_suffix", "")

    def one_name_to_group(value: str) -> str:
        parts = str(value).split("/")
        key = parts[segment_idx] if len(parts) > segment_idx else Path(value).name
        if remove_suffix and key.endswith(remove_suffix):
            key = key[: -len(remove_suffix)]
        return key

    return names.map(one_name_to_group)


def split_dataset(
    dataset: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = dataset[cfg["dataset"]["score_column"]]
    X = dataset.drop(columns=[cfg["dataset"]["name_column"], cfg["dataset"]["score_column"]])
    X = X.apply(pd.to_numeric, errors="raise")

    groups = build_group_keys(dataset[cfg["dataset"]["name_column"]], cfg)
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"], random_state=cfg["seed"])
    train_idx, test_idx = next(splitter.split(dataset, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    if cfg["scale_features"]:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test


def _config_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def infer_feature_categories_from_pipeline_config(cfg: dict[str, Any]) -> dict[str, list[str]]:
    features_cfg = cfg.get("features", {})
    if not isinstance(features_cfg, dict):
        return {}

    categories: dict[str, list[str]] = {
        "nr_metrics": [],
        "fr_metrics": [],
        "timm_prefixes": [],
    }

    sources: list[dict[str, Any]] = []
    common = features_cfg.get("common", {})
    if isinstance(common, dict):
        sources.append(common)

    groups = features_cfg.get("groups", {})
    if isinstance(groups, dict):
        sources.extend(group for group in groups.values() if isinstance(group, dict))
    elif isinstance(groups, list):
        sources.extend(group for group in groups if isinstance(group, dict))

    for source in sources:
        categories["nr_metrics"].extend(_config_list(source.get("nr_metrics")))
        categories["fr_metrics"].extend(_config_list(source.get("fr_metrics")))

        timm_encoders = source.get("timm_encoders")
        if isinstance(timm_encoders, dict):
            categories["timm_prefixes"].extend(str(name) for name in timm_encoders)
        else:
            for spec in _config_list(timm_encoders):
                categories["timm_prefixes"].append(spec.split("=", 1)[0].strip())

    return {
        key: sorted(set(value), key=str.lower)
        for key, value in categories.items()
        if value
    }


def configured_feature_categories(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    categories = {
        "nr_metrics": list(DEFAULT_NR_METRICS),
        "fr_metrics": list(DEFAULT_FR_METRICS),
        "timm_prefixes": [],
    }
    if not cfg:
        return categories

    for source in [cfg.get("feature_categories", {}), cfg.get("features", {})]:
        if not isinstance(source, dict):
            continue
        for key in categories:
            if key in source:
                categories[key].extend(_config_list(source[key]))

    categories["nr_metrics"] = sorted(set(categories["nr_metrics"]), key=str.lower)
    categories["fr_metrics"] = sorted(set(categories["fr_metrics"]), key=str.lower)
    categories["timm_prefixes"] = sorted(set(categories["timm_prefixes"]), key=str.lower)
    return categories


def feature_family(feature_name: str, cfg: dict[str, Any] | None = None) -> str:
    feature = feature_name.lower()
    categories = configured_feature_categories(cfg)
    nr_metrics = [metric.lower() for metric in categories["nr_metrics"]]
    fr_metrics = [metric.lower() for metric in categories["fr_metrics"]]
    timm_prefixes = [prefix.lower().rstrip("_") for prefix in categories["timm_prefixes"]]

    if feature in nr_metrics:
        return "NR"
    if any(feature == prefix or feature.startswith(prefix + "_") for prefix in timm_prefixes):
        return "Timm"
    if any(
        feature == metric or feature.startswith(metric + "_")
        for metric in fr_metrics
    ):
        return "FR"
    if feature.startswith("vgg_"):
        return "VGG"
    if feature.startswith("resnet_"):
        return "ResNet"
    if feature in {"content_fidelity", "perceptual_enhancement", "final_rr_score"}:
        return "SigLIP"
    if feature in {"min", "max", "mean", "median", "std", "p05", "p95", "area00", "area05", "area075"}:
        return "Stats"
    return "Other"


def importance_palette() -> dict[str, str]:
    return {
        "NR": "#ff6150",
        "FR": "#f8aa4b",
        "Timm": "#8a63d2",
        "VGG": "#54d2d2",
        "ResNet": "#0e4a95",
        "SigLIP": "#5255ea",
        "Stats": "#5bea52",
        "Other": "#000000",
    }


def importance_legend_labels(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    labels = {
        "NR": "NR metrics",
        "FR": "FR metrics",
        "Timm": "timm embeddings",
        "VGG": "VGG features",
        "ResNet": "ResNet features",
        "SigLIP": "SigLIP features",
        "Stats": "Artifact statistics",
        "Other": "Other",
    }
    pretty_cfg = cfg.get("pretty_names", {}) if cfg else {}
    family_names = pretty_cfg.get("families", {}) if isinstance(pretty_cfg, dict) else {}
    if isinstance(family_names, dict):
        labels.update({str(key): str(value) for key, value in family_names.items()})
    return labels


def model_display_name(model_name: str, cfg: dict[str, Any] | None = None) -> str:
    return get_pretty_feature(model_name, cfg)


def _missing_optional(package_name: str, extra_name: str) -> ImportError:
    return ImportError(
        f"Optional dependency '{package_name}' is required for this enabled model. "
        f"Install it with `pip install -e .[{extra_name}]` or disable the model in the config."
    )


def plot_rc_params(cfg: dict[str, Any]) -> dict[str, Any]:
    font_size = cfg.get("plot", {}).get("font_size")
    return {"font.size": font_size} if font_size is not None else {}


def plot_enabled(cfg: dict[str, Any], name: str) -> bool:
    enabled = cfg.get("plot", {}).get("enabled", {})
    if isinstance(enabled, bool):
        return enabled
    if isinstance(enabled, dict):
        return bool(enabled.get(name, True))
    return True


def save_plot(fig: plt.Figure, out_path: Path, cfg: dict[str, Any]) -> None:
    savefig_kwargs = {}
    plot_cfg = cfg.get("plot", {})
    dpi = cfg.get("plot", {}).get("dpi")
    if dpi is not None:
        savefig_kwargs["dpi"] = dpi
    bbox_inches = plot_cfg.get("bbox_inches", "tight")
    if bbox_inches:
        savefig_kwargs["bbox_inches"] = bbox_inches
    pad_inches = plot_cfg.get("pad_inches", 0.1)
    if pad_inches is not None:
        savefig_kwargs["pad_inches"] = pad_inches
    fig.savefig(out_path, **savefig_kwargs)
    if cfg.get("plot", {}).get("save_svg", False):
        fig.savefig(out_path.with_suffix(".svg"), **savefig_kwargs)


def model_params(cfg: dict[str, Any], model_name: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(defaults or {})
    params.update(cfg["models"].get(model_name, {}).get("params", {}))
    return params


def init_xgb_model(cfg: dict[str, Any], model_name: str) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise _missing_optional("xgboost", "regressors") from exc

    return xgb.XGBRegressor(**model_params(cfg, model_name, {"random_state": cfg["seed"], "verbosity": 0}))


def init_catboost_model(cfg: dict[str, Any], model_name: str) -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise _missing_optional("catboost", "regressors") from exc

    return CatBoostRegressor(**model_params(cfg, model_name, {"random_state": cfg["seed"], "verbose": 0}))


def init_lgbm_model(cfg: dict[str, Any], model_name: str) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise _missing_optional("lightgbm", "regressors") from exc

    return LGBMRegressor(**model_params(cfg, model_name, {"random_state": cfg["seed"], "verbosity": -1}))


MODEL_FACTORIES = {
    "randomforest": lambda cfg, name: RandomForestRegressor(
        **model_params(cfg, name, {"random_state": cfg["seed"]})
    ),
    "gradientboosting": lambda cfg, name: GradientBoostingRegressor(
        **model_params(cfg, name, {"random_state": cfg["seed"]})
    ),
    "gradientboost": lambda cfg, name: GradientBoostingRegressor(
        **model_params(cfg, name, {"random_state": cfg["seed"]})
    ),
    "xgb": init_xgb_model,
    "catboost": init_catboost_model,
    "lgbm": init_lgbm_model,
    "lightgbm": init_lgbm_model,
    "linear": lambda cfg, name: LinearRegression(**model_params(cfg, name)),
    "ridge": lambda cfg, name: Ridge(**model_params(cfg, name, {"random_state": cfg["seed"]})),
    "lasso": lambda cfg, name: Lasso(
        **model_params(cfg, name, {"random_state": cfg["seed"], "alpha": 0.001, "max_iter": 10000})
    ),
    "elasticnet": lambda cfg, name: ElasticNet(
        **model_params(
            cfg,
            name,
            {"random_state": cfg["seed"], "alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10000},
        )
    ),
    "svr": lambda cfg, name: SVR(**model_params(cfg, name)),
    "linear_svr": lambda cfg, name: LinearSVR(
        **model_params(cfg, name, {"random_state": cfg["seed"], "max_iter": 10000})
    ),
    "mlp": lambda cfg, name: MLPRegressor(
        **model_params(cfg, name, {"random_state": cfg["seed"], "max_iter": 1000})
    ),
}


def init_models(cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    models_cfg = cfg["models"]
    initialized: list[tuple[str, Any]] = []

    for model_name, model_cfg in models_cfg.items():
        if not isinstance(model_cfg, dict) or not model_cfg.get("enabled", False):
            continue
        factory = MODEL_FACTORIES.get(model_name)
        if factory is None:
            supported = ", ".join(sorted(MODEL_FACTORIES))
            raise ValueError(f"Unsupported enabled model '{model_name}'. Supported models: {supported}")
        initialized.append((model_name, factory(cfg, model_name)))

    if not initialized:
        raise ValueError("No models are enabled in config['models']")

    return initialized


def native_feature_importances(model: Any, n_features: int) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim > 1:
            values = np.mean(np.abs(values), axis=0)
        else:
            values = np.abs(values)
    elif hasattr(model, "coefs_") and getattr(model, "coefs_", None):
        first_layer = np.asarray(model.coefs_[0], dtype=float)
        values = np.mean(np.abs(first_layer), axis=1)
    else:
        return None

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != n_features:
        raise ValueError(f"importance vector has {values.size} values for {n_features} features")
    return values


def compute_plot_importances(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.Series, pd.Series | None] | None:
    native_values = None
    try:
        native_values = native_feature_importances(model, X_test.shape[1])
    except Exception as exc:
        warnings.warn(
            f"Could not read native importances for {model_display_name(model_name, cfg)}: {exc}. "
            "Using permutation importances instead.",
            stacklevel=2,
        )

    permutation = None
    try:
        permutation = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=cfg["permutation_repeats"],
            random_state=cfg["seed"],
            n_jobs=2,
        )
    except Exception as exc:
        if native_values is None:
            warnings.warn(
                f"Skipping importance plot for {model_display_name(model_name, cfg)}: "
                f"permutation importances failed: {exc}",
                stacklevel=2,
            )
            return None
        warnings.warn(
            f"Permutation uncertainty failed for {model_display_name(model_name, cfg)}: {exc}",
            stacklevel=2,
        )

    if native_values is None:
        importances = pd.Series(np.abs(permutation.importances_mean), index=X_test.columns)
    else:
        importances = pd.Series(native_values, index=X_test.columns)

    importances = importances.sort_values(ascending=True)
    if permutation is None:
        return importances, None

    yerr = pd.Series(permutation.importances_std, index=X_test.columns).reindex(importances.index)
    return importances, yerr


def plot_importance(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    computed = compute_plot_importances(model_name, model, X_test, y_test, cfg)
    if computed is None:
        return None
    importances, perm_std = computed

    palette = importance_palette()
    colors = [palette[feature_family(name, cfg)] for name in importances.index]
    display_importances = importances.copy()
    display_importances.index = get_pretty_labels(importances.index, cfg)

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["importance_figsize"]))
        yerr = perm_std.to_numpy() if perm_std is not None else None
        display_importances.plot.barh(yerr=yerr, ax=ax, color=colors)
        ax.set_title(f"{model_display_name(model_name, cfg)}\nFeature Importances", pad=10)
        ax.set_xlabel("Importance")
        fig.tight_layout()

    out_path = out_dir / f"importance_{model_name}.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def _normalize_shap_values(shap_values: Any, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            shap_values = shap_values[0]
        else:
            shap_values = np.mean(np.asarray(shap_values, dtype=float), axis=0)

    shap_arr = np.asarray(shap_values, dtype=float)
    n_samples, n_features = X_test.shape

    if shap_arr.ndim == 3:
        if shap_arr.shape[:2] == (n_samples, n_features):
            shap_arr = shap_arr[:, :, 0] if shap_arr.shape[2] == 1 else np.mean(shap_arr, axis=2)
        elif shap_arr.shape[1:] == (n_samples, n_features):
            shap_arr = shap_arr[0] if shap_arr.shape[0] == 1 else np.mean(shap_arr, axis=0)

    if shap_arr.ndim != 2 or shap_arr.shape != (n_samples, n_features):
        raise ValueError(
            "Unexpected SHAP values shape: "
            f"{shap_arr.shape}; expected {(n_samples, n_features)} for {n_samples} samples "
            f"and {n_features} features"
        )

    return shap_arr


def _xgboost_shap_values(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise _missing_optional("xgboost", "regressors") from exc

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
    contributions = np.asarray(booster.predict(dmatrix, pred_contribs=True), dtype=float)
    if contributions.ndim == 3:
        contributions = contributions[:, :, 0] if contributions.shape[2] == 1 else np.mean(contributions, axis=2)

    expected_shape = (X_test.shape[0], X_test.shape[1] + 1)
    if contributions.shape != expected_shape:
        raise ValueError(
            "Unexpected XGBoost SHAP contribution shape: "
            f"{contributions.shape}; expected {expected_shape}"
        )

    return contributions[:, :-1]


def shap_sample_frame(X: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(X) <= max_samples:
        return X
    return X.sample(n=max_samples, random_state=seed).sort_index()


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _kernel_shap_values(shap: Any, model: Any, X_test: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    plot_cfg = cfg.get("plot", {})
    background_samples = plot_cfg.get("shap_kernel_background_samples", 32)
    max_samples = plot_cfg.get("shap_kernel_max_samples", 100)
    nsamples = plot_cfg.get("shap_kernel_nsamples", "auto")

    background = shap_sample_frame(X_test, optional_int(background_samples), cfg["seed"])
    explain_data = shap_sample_frame(X_test, optional_int(max_samples), cfg["seed"])

    def predict_fn(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=X_test.columns)
        return np.asarray(model.predict(frame), dtype=float).reshape(-1)

    explainer = shap.KernelExplainer(predict_fn, background)
    kwargs = {}
    if nsamples is not None:
        kwargs["nsamples"] = nsamples
    values = explainer.shap_values(explain_data, **kwargs)
    return _normalize_shap_values(values, explain_data)


def compute_model_shap_values(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    if model_name == "xgb":
        return _xgboost_shap_values(model, X_test)

    try:
        import shap
    except ImportError:
        warnings.warn(
            "Optional dependency 'shap' is required for SHAP plots. "
            "Install it with `pip install -e .[regressors]` or set "
            "`plot.enabled.shap` to false in the config.",
            stacklevel=2,
        )
        return None

    try:
        explainer = shap.TreeExplainer(model)
        return _normalize_shap_values(explainer.shap_values(X_test), X_test)
    except Exception as exc:
        if not cfg.get("plot", {}).get("shap_kernel_fallback", True):
            raise
        warnings.warn(
            f"TreeExplainer failed for {model_name}: {exc}. Falling back to KernelExplainer.",
            stacklevel=2,
        )
        return _kernel_shap_values(shap, model, X_test, cfg)


def plot_shap_importance(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    try:
        shap_arr = compute_model_shap_values(model_name, model, X_test, cfg)
    except Exception as exc:
        warnings.warn(
            f"Skipping SHAP plot for {model_display_name(model_name, cfg)}: {exc}",
            stacklevel=2,
        )
        return None
    if shap_arr is None:
        return None

    shap_values_path = out_dir / f"shap_values_{model_name}.csv"
    pd.DataFrame(shap_arr, columns=X_test.columns).to_csv(shap_values_path, index=False)

    mean_abs = pd.Series(np.abs(shap_arr).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    mean_abs_df = pd.DataFrame(
        {
            "feature": mean_abs.index,
            "pretty_feature": get_pretty_labels(mean_abs.index, cfg),
            "family": [feature_family(name, cfg) for name in mean_abs.index],
            "mean_abs_shap": mean_abs.values,
        }
    )
    mean_abs_df.to_csv(out_dir / f"shap_mean_abs_{model_name}.csv", index=False)

    plot_values = mean_abs.sort_values(ascending=True)
    palette = importance_palette()
    colors = [palette[feature_family(name, cfg)] for name in plot_values.index]
    pretty_labels = get_pretty_labels(plot_values.index, cfg)
    max_value = float(plot_values.max()) if len(plot_values) else 0.0
    default_height = max(5.0, 0.45 * len(plot_values))
    figsize = tuple(cfg.get("plot", {}).get("shap_figsize", [8, default_height]))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(plot_values)), plot_values.values, color=colors, height=0.65)
        ax.set_yticks(range(len(plot_values)))
        ax.set_yticklabels(pretty_labels)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_display_name(model_name, cfg)}\nSHAP Feature Importance", pad=10)
        if max_value > 0:
            ax.set_xlim(0, max_value * 1.15)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

    out_path = out_dir / f"shap_importance_{model_name}.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def plot_all_importances(
    importance_paths: dict[str, str | None],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    valid = []
    for model_name, path in importance_paths.items():
        if path is None:
            continue
        existing_path = Path(path)
        if existing_path.exists():
            valid.append((model_name, existing_path))

    if not valid:
        return None

    images = [plt.imread(path) for _, path in valid]
    single_w, single_h = tuple(cfg["plot"]["importance_figsize"])
    max_cols = max(1, int(cfg.get("plot", {}).get("aggregate_max_columns", 3)))
    n_cols = min(len(images), max_cols)
    n_rows = math.ceil(len(images) / n_cols)
    with plt.rc_context(plot_rc_params(cfg)):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(single_w * n_cols, single_h * n_rows))
        axes_flat = np.asarray(axes).reshape(-1)

        for ax, (_, _), image in zip(axes_flat, valid, images, strict=False):
            ax.imshow(image)
            ax.axis("off")
        for ax in axes_flat[len(images):]:
            ax.axis("off")

        palette = importance_palette()
        labels = importance_legend_labels(cfg)
        handles = [mpatches.Patch(color=palette[key], label=labels[key]) for key in labels]

        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5))
        fig.tight_layout(rect=(0, 0, 0.9, 1))

    out_path = out_dir / "all_models_importances.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def plot_all_shap_importances(
    shap_paths: dict[str, str | None],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    valid = []
    for model_name, path in shap_paths.items():
        if path is None:
            continue
        existing_path = Path(path)
        if existing_path.exists():
            valid.append((model_name, existing_path))

    if not valid:
        return None

    images = [plt.imread(path) for _, path in valid]
    single_w, single_h = tuple(cfg.get("plot", {}).get("shap_figsize", cfg["plot"]["importance_figsize"]))
    max_cols = max(1, int(cfg.get("plot", {}).get("aggregate_max_columns", 3)))
    n_cols = min(len(images), max_cols)
    n_rows = math.ceil(len(images) / n_cols)
    with plt.rc_context(plot_rc_params(cfg)):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(single_w * n_cols, single_h * n_rows))
        axes_flat = np.asarray(axes).reshape(-1)

        for ax, (_, _), image in zip(axes_flat, valid, images, strict=False):
            ax.imshow(image)
            ax.axis("off")
        for ax in axes_flat[len(images):]:
            ax.axis("off")

        palette = importance_palette()
        labels = importance_legend_labels(cfg)
        handles = [mpatches.Patch(color=palette[key], label=labels[key]) for key in labels]

        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5))
        fig.tight_layout(rect=(0, 0, 0.9, 1))

    out_path = out_dir / "all_models_shap_importances.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def plot_correlations(
    results_df: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
    filename: str = "correlations.png",
    title: str = "Regressor and Metric Correlation Scores",
) -> Path:
    df = results_df.sort_values("srcc", ascending=False).reset_index(drop=True)

    bar_width = 0.18
    x = np.arange(len(df))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["correlation_figsize"]))
        ax.bar(x - bar_width / 2, df["plcc"], width=bar_width, label="PLCC", color="#845ec2")
        ax.bar(x + bar_width / 2, df["srcc"], width=bar_width, label="SRCC", color="#00c9a7")
        ax.set_xticks(x)
        ax.set_xticklabels(get_pretty_labels(df["model"], cfg), rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.tight_layout()

    out_path = out_dir / filename
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def plot_prediction_scatter(
    predictions_by_model: dict[str, pd.DataFrame],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if not predictions_by_model:
        return None

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg.get("plot", {}).get("scatter_figsize", [8.5, 5.5])))
        ax.plot([0, 1], [0, 1], color="#9aa3ad", linestyle="--", linewidth=1.2)

        for model_name, predictions in predictions_by_model.items():
            ax.scatter(
                predictions["mos"],
                predictions["prediction"],
                s=cfg.get("plot", {}).get("scatter_point_size", 100),
                alpha=0.78,
                edgecolors="white",
                linewidths=0.6,
                label=model_display_name(model_name, cfg),
            )

        ax.set_xlabel("MOS")
        ax.set_ylabel("Prediction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", frameon=True, framealpha=0.88, edgecolor="#d6dbe1")
        fig.tight_layout()

    out_path = out_dir / "mos_prediction_scatter_all_regressors.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def compute_feature_correlations(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    target = pd.to_numeric(y_test.reset_index(drop=True), errors="coerce")
    rows = []

    for feature_name in X_test.columns:
        values = pd.to_numeric(X_test[feature_name].reset_index(drop=True), errors="coerce")
        valid = target.notna() & values.notna()
        if valid.any():
            plcc, srcc = safe_corr(target[valid], values[valid])
        else:
            plcc, srcc = np.nan, np.nan

        rows.append(
            {
                "feature": feature_name,
                "family": feature_family(feature_name, cfg),
                "plcc": plcc,
                "srcc": srcc,
                "abs_plcc": abs(plcc) if not np.isnan(plcc) else np.nan,
                "abs_srcc": abs(srcc) if not np.isnan(srcc) else np.nan,
            }
        )

    mean_values = X_test.apply(pd.to_numeric, errors="coerce").mean(axis=1).reset_index(drop=True)
    valid = target.notna() & mean_values.notna()
    if valid.any():
        plcc, srcc = safe_corr(target[valid], mean_values[valid])
    else:
        plcc, srcc = np.nan, np.nan
    rows.append(
        {
            "feature": "mean_features",
            "family": "Mean",
            "plcc": plcc,
            "srcc": srcc,
            "abs_plcc": abs(plcc) if not np.isnan(plcc) else np.nan,
            "abs_srcc": abs(srcc) if not np.isnan(srcc) else np.nan,
        }
    )

    median_values = X_test.apply(pd.to_numeric, errors="coerce").median(axis=1).reset_index(drop=True)
    valid = target.notna() & median_values.notna()
    if valid.any():
        plcc, srcc = safe_corr(target[valid], median_values[valid])
    else:
        plcc, srcc = np.nan, np.nan
    rows.append(
        {
            "feature": "median_features",
            "family": "Median",
            "plcc": plcc,
            "srcc": srcc,
            "abs_plcc": abs(plcc) if not np.isnan(plcc) else np.nan,
            "abs_srcc": abs(srcc) if not np.isnan(srcc) else np.nan,
        }
    )

    return pd.DataFrame(rows).sort_values("abs_srcc", ascending=False).reset_index(drop=True)


def plot_feature_correlations(
    feature_correlations: pd.DataFrame,
    results_df: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if feature_correlations.empty:
        return None

    feature_rows = feature_correlations.rename(columns={"feature": "name", "family": "group"}).copy()
    feature_rows["kind"] = "feature"
    feature_rows.loc[feature_rows["group"] == "Mean", "kind"] = "regressor"
    feature_rows.loc[feature_rows["group"] == "Median", "kind"] = "regressor"

    if "source" in results_df.columns:
        regressor_results = results_df[results_df["source"] == "regressor"].copy()
    else:
        regressor_results = pd.DataFrame()

    if not regressor_results.empty:
        regressor_rows = regressor_results.rename(columns={"model": "name"}).copy()
        regressor_rows["group"] = "Regressor"
        regressor_rows["kind"] = "regressor"
        plot_df = pd.concat(
            [
                feature_rows[["name", "group", "kind", "plcc", "srcc"]],
                regressor_rows[["name", "group", "kind", "plcc", "srcc"]],
            ],
            ignore_index=True,
        )
    else:
        plot_df = feature_rows[["name", "group", "kind", "plcc", "srcc"]]

    plot_df["abs_srcc"] = plot_df["srcc"].abs()
    # plot_df = plot_df.sort_values(["kind", "abs_srcc"], ascending=[True, True]).reset_index(drop=True)
    plot_df = plot_df.sort_values(["kind", "srcc"], ascending=[True, True]).reset_index(drop=True)

    # plot_df = plot_df[plot_df["srcc"] > 0]
    plot_df["name"] = get_pretty_labels(plot_df["name"], cfg)

    y = np.arange(len(plot_df))
    bar_height = 0.36
    is_regressor = plot_df["kind"] == "regressor"
    is_mean_features = plot_df["kind"] == "mean_features"
    is_median_features = plot_df["kind"] == "median_features"
    plcc_colors = np.select(
        [is_regressor, is_mean_features, is_median_features],
        ["#845ec2", "#845ec2", "#845ec2"],
        default="#4d5a68",
    )
    srcc_colors = np.select(
        [is_regressor, is_mean_features, is_median_features],
        ["#00c9a7", "#00c9a7", "#00c9a7"],
        default="#b8c0cc",
    )

    default_height = max(6, 0.5 * len(plot_df))
    figsize = tuple(cfg.get("plot", {}).get("feature_correlation_figsize", [9, default_height]))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y + bar_height / 2, plot_df["plcc"], height=bar_height, color=plcc_colors)
        ax.barh(y - bar_height / 2, plot_df["srcc"], height=bar_height, color=srcc_colors)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["name"].tolist())
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_xlabel("Correlation")
        ax.set_title("Feature and Predictor Correlations")

        handles = [
            mpatches.Patch(color="#4d5a68", label="Feature PLCC"),
            mpatches.Patch(color="#b8c0cc", label="Feature SRCC"),
            mpatches.Patch(color="#845ec2", label="Predictor PLCC"),
            mpatches.Patch(color="#00c9a7", label="Predictor SRCC"),
        ]
        ax.legend(handles=handles, loc="lower right")
        fig.tight_layout()

    out_path = out_dir / "feature_correlations.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def feature_matrix_for_cross_correlation(X: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    numeric = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_features = cfg.get("plot", {}).get("max_feature_correlation_matrix_features", 300)
    if max_features is None or numeric.shape[1] <= int(max_features):
        return numeric

    variances = numeric.var(axis=0, skipna=True).sort_values(ascending=False)
    selected = variances.head(int(max_features)).index.tolist()
    return numeric[selected]


def compute_feature_cross_correlations(X: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    matrix = feature_matrix_for_cross_correlation(X, cfg)
    return matrix.corr(method="pearson")


def plot_feature_cross_correlation_matrix(
    cross_correlations: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if cross_correlations.empty:
        return None

    n_features = len(cross_correlations)
    default_size = min(max(7.0, 0.34 * n_features), 24.0)
    figsize = tuple(cfg.get("plot", {}).get("feature_correlation_matrix_figsize", [default_size, default_size]))
    label_limit = int(cfg.get("plot", {}).get("max_feature_correlation_matrix_labels", 45))
    show_labels = n_features <= label_limit
    label_font_size = cfg.get("plot", {}).get(
        "feature_correlation_matrix_label_font_size",
        min(10, plt.rcParams["font.size"]),
    )

    values = np.ma.masked_invalid(cross_correlations.to_numpy(dtype=float))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, interpolation="nearest")
        # title = "Feature Cross-Correlation Matrix"
        # if show_labels:
        #     fig.suptitle(title, y=0.98)
        # else:
        #     ax.set_title(title)

        if show_labels:
            ticks = np.arange(n_features)
            labels = get_pretty_labels(cross_correlations.columns, cfg)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=label_font_size)
            ax.set_yticklabels(labels, fontsize=label_font_size)
            ax.tick_params(
                axis="x",
                which="both",
                top=True,
                bottom=True,
                labeltop=True,
                labelbottom=True,
                pad=2,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=True,
                labelleft=True,
                labelright=True,
                pad=2,
            )
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_rotation(90)
                tick.label1.set_ha("center")
                tick.label1.set_va("top")
                tick.label1.set_fontsize(label_font_size)
                tick.label2.set_rotation(90)
                tick.label2.set_ha("center")
                tick.label2.set_va("bottom")
                tick.label2.set_fontsize(label_font_size)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_ha("right")
                tick.label2.set_ha("left")
                tick.label1.set_fontsize(label_font_size)
                tick.label2.set_fontsize(label_font_size)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{n_features} features")
            ax.set_ylabel(f"{n_features} features")

        if show_labels:
            fig.subplots_adjust(left=0.18, right=0.70, bottom=0.22, top=0.74)
            cbar_ax = fig.add_axes([0.88, 0.22, 0.025, 0.52])
            cbar = fig.colorbar(image, cax=cbar_ax)
        else:
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
        cbar.set_label("Pearson correlation")

    out_path = out_dir / "feature_cross_correlation_matrix.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def analysis_enabled(cfg: dict[str, Any], section: str) -> bool:
    section_cfg = cfg.get("analysis", {}).get(section, {})
    return bool(section_cfg.get("enabled", False)) if isinstance(section_cfg, dict) else False


def resolve_feature_specs(specs: Any, columns: pd.Index) -> list[str]:
    if specs is None or specs == []:
        return list(columns)
    requested = _config_list(specs)
    selected: list[str] = []
    for spec in requested:
        matches = [column for column in columns if column == spec or fnmatch(column, spec)]
        if not matches and spec.endswith("_"):
            matches = [column for column in columns if column.startswith(spec)]
        if not matches and spec in columns:
            matches = [spec]
        selected.extend(matches)
    return list(dict.fromkeys(selected))


def compute_feature_outliers(
    X: pd.DataFrame,
    names: pd.Series,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    numeric = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    means = numeric.mean(axis=0, skipna=True)
    stds = numeric.std(axis=0, skipna=True).replace(0, np.nan)
    z = ((numeric - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_abs_z = z.abs().max(axis=1)
    mean_abs_z = z.abs().mean(axis=1)
    euclidean_z = np.sqrt((z**2).sum(axis=1))
    strongest_feature = z.abs().idxmax(axis=1)

    threshold = float(cfg.get("analysis", {}).get("outliers", {}).get("feature_z_threshold", 3.0))
    return pd.DataFrame(
        {
            "name": names.reset_index(drop=True),
            "max_abs_z": max_abs_z.reset_index(drop=True),
            "mean_abs_z": mean_abs_z.reset_index(drop=True),
            "euclidean_z": pd.Series(euclidean_z).reset_index(drop=True),
            "strongest_feature": strongest_feature.reset_index(drop=True),
            "is_outlier": max_abs_z.reset_index(drop=True) >= threshold,
        }
    ).sort_values("max_abs_z", ascending=False).reset_index(drop=True)


def compute_prediction_outliers(predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions_by_model:
        return pd.DataFrame()

    base = None
    residual_columns = []
    prediction_columns = []
    for model_name, predictions in predictions_by_model.items():
        current = predictions[["name", "mos", "prediction"]].copy()
        pred_col = f"prediction_{model_name}"
        residual_col = f"abs_error_{model_name}"
        current = current.rename(columns={"prediction": pred_col})
        current[residual_col] = (current[pred_col] - current["mos"]).abs()
        residual_columns.append(residual_col)
        prediction_columns.append(pred_col)
        base = current if base is None else base.merge(current.drop(columns=["mos"]), on="name", how="outer")

    if base is None:
        return pd.DataFrame()

    base["max_abs_error"] = base[residual_columns].max(axis=1)
    base["mean_abs_error"] = base[residual_columns].mean(axis=1)
    base["prediction_std"] = base[prediction_columns].std(axis=1)
    base["worst_model"] = base[residual_columns].idxmax(axis=1).str.replace("abs_error_", "", regex=False)
    return base.sort_values("max_abs_error", ascending=False).reset_index(drop=True)


def plot_outlier_scores(
    df: pd.DataFrame,
    score_column: str,
    label_column: str,
    title: str,
    out_path: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if df.empty or score_column not in df.columns:
        return None

    top_n = int(cfg.get("analysis", {}).get("outliers", {}).get("top_n", 25))
    plot_df = df.head(top_n).sort_values(score_column, ascending=True)
    labels = plot_df[label_column].astype(str).tolist()
    default_height = max(5, 0.35 * len(plot_df))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg.get("plot", {}).get("outlier_figsize", [10, default_height])))
        ax.barh(np.arange(len(plot_df)), plot_df[score_column], color="#536dfe")
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(score_column.replace("_", " "))
        ax.set_title(title)
        fig.tight_layout()

    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def save_outlier_analysis(
    X_all: pd.DataFrame,
    names_all: pd.Series,
    predictions_by_model: dict[str, pd.DataFrame],
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str | None]:
    outlier_dir = ensure_dir(out_dir / "outliers")
    paths: dict[str, str | None] = {}

    feature_outliers = compute_feature_outliers(X_all, names_all, cfg)
    feature_csv = outlier_dir / "feature_outliers.csv"
    feature_outliers.to_csv(feature_csv, index=False)
    paths["feature_outliers_csv"] = str(feature_csv)
    feature_plot = plot_outlier_scores(
        feature_outliers,
        "max_abs_z",
        "name",
        "Feature-Space Outliers",
        outlier_dir / "feature_outliers.png",
        cfg,
    )
    paths["feature_outliers_plot"] = str(feature_plot) if feature_plot else None

    prediction_outliers = compute_prediction_outliers(predictions_by_model)
    if not prediction_outliers.empty:
        prediction_csv = outlier_dir / "prediction_outliers.csv"
        prediction_outliers.to_csv(prediction_csv, index=False)
        paths["prediction_outliers_csv"] = str(prediction_csv)
        prediction_plot = plot_outlier_scores(
            prediction_outliers,
            "max_abs_error",
            "name",
            "Prediction Outliers",
            outlier_dir / "prediction_outliers.png",
            cfg,
        )
        paths["prediction_outliers_plot"] = str(prediction_plot) if prediction_plot else None

    return paths


def binned_target(y: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(y, errors="coerce")
    try:
        return pd.qcut(numeric, q=bins, duplicates="drop", labels=False)
    except ValueError:
        return pd.Series(np.nan, index=y.index)


def fisher_scores(X: pd.DataFrame, y_bins: pd.Series) -> pd.Series:
    scores = {}
    valid_classes = y_bins.dropna().unique()
    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce")
        valid = values.notna() & y_bins.notna()
        if valid.sum() < 2:
            scores[feature] = np.nan
            continue
        overall_mean = values[valid].mean()
        between = 0.0
        within = 0.0
        for klass in valid_classes:
            mask = valid & (y_bins == klass)
            if not mask.any():
                continue
            class_values = values[mask]
            between += len(class_values) * float((class_values.mean() - overall_mean) ** 2)
            within += len(class_values) * float(class_values.var(ddof=0))
        scores[feature] = between / within if within > 0 else np.nan
    return pd.Series(scores)


def compute_feature_analysis_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    analysis_cfg = cfg.get("analysis", {}).get("feature_metrics", {})
    selected = resolve_feature_specs(analysis_cfg.get("features"), X.columns)
    X_selected = X[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_selected = X_selected.fillna(X_selected.median(axis=0)).fillna(0.0)
    y_numeric = pd.to_numeric(y, errors="coerce")

    rows = pd.DataFrame({"feature": selected})
    rows["pretty_feature"] = get_pretty_labels(selected, cfg)
    rows["family"] = [feature_family(feature, cfg) for feature in selected]

    try:
        rows["mutual_info"] = mutual_info_regression(
            X_selected,
            y_numeric,
            random_state=cfg["seed"],
        )
    except Exception as exc:
        warnings.warn(f"Mutual information feature analysis failed: {exc}", stacklevel=2)
        rows["mutual_info"] = np.nan

    try:
        f_values, p_values = f_regression(X_selected, y_numeric)
        rows["f_score"] = f_values
        rows["f_pvalue"] = p_values
    except Exception as exc:
        warnings.warn(f"F-score feature analysis failed: {exc}", stacklevel=2)
        rows["f_score"] = np.nan
        rows["f_pvalue"] = np.nan

    bins = int(analysis_cfg.get("target_bins", 5))
    y_bins = binned_target(y_numeric, bins)
    try:
        nonnegative = pd.DataFrame(
            MinMaxScaler().fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index,
        )
        chi_values, chi_pvalues = chi2(nonnegative, y_bins)
        rows["chi2"] = chi_values
        rows["chi2_pvalue"] = chi_pvalues
    except Exception as exc:
        warnings.warn(f"Chi-square feature analysis failed: {exc}", stacklevel=2)
        rows["chi2"] = np.nan
        rows["chi2_pvalue"] = np.nan

    rows["fisher_score"] = fisher_scores(X_selected, y_bins).reindex(selected).to_numpy()
    rank_by = str(analysis_cfg.get("rank_by", "mutual_info"))
    if rank_by not in rows.columns:
        rank_by = "mutual_info"
    rows["rank_score"] = rows[rank_by]
    return rows.sort_values("rank_score", ascending=False).reset_index(drop=True)


def plot_feature_analysis_metrics(metrics_df: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> Path | None:
    if metrics_df.empty:
        return None

    analysis_cfg = cfg.get("analysis", {}).get("feature_metrics", {})
    rank_by = str(analysis_cfg.get("rank_by", "mutual_info"))
    if rank_by not in metrics_df.columns:
        rank_by = "mutual_info"
    top_n = int(analysis_cfg.get("top_n", 40))
    plot_df = metrics_df.head(top_n).sort_values(rank_by, ascending=True)
    default_height = max(6, 0.35 * len(plot_df))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg.get("plot", {}).get("feature_metric_figsize", [10, default_height])))
        ax.barh(np.arange(len(plot_df)), plot_df[rank_by], color="#0077b6")
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(plot_df["pretty_feature"].tolist())
        ax.set_xlabel(rank_by.replace("_", " "))
        ax.set_title("Feature Analysis Ranking")
        fig.tight_layout()

    out_path = out_dir / "feature_analysis_metrics.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def save_feature_analysis_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str | None]:
    feature_dir = ensure_dir(out_dir / "feature_analysis")
    metrics = compute_feature_analysis_metrics(X_train, y_train, cfg)
    metrics_csv = feature_dir / "feature_analysis_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    plot_path = plot_feature_analysis_metrics(metrics, feature_dir, cfg)
    return {
        "feature_analysis_metrics_csv": str(metrics_csv),
        "feature_analysis_metrics_plot": str(plot_path) if plot_path else None,
    }


def init_analysis_model(cfg: dict[str, Any], model_name: str, params: dict[str, Any] | None = None) -> Any:
    local_cfg = deepcopy(cfg)
    local_cfg["models"] = {model_name: {"enabled": True, "params": params or {}}}
    return init_models(local_cfg)[0][1]


def evaluate_feature_subset(
    cfg: dict[str, Any],
    model_name: str,
    model_params_override: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
) -> tuple[float, float]:
    if not features:
        return np.nan, np.nan
    model = init_analysis_model(cfg, model_name, model_params_override)
    model.fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    return safe_corr(y_test, pred)


def feature_selection_score(plcc: float, srcc: float, metric: str) -> float:
    value = srcc if metric == "srcc" else plcc
    return value if not np.isnan(value) else -np.inf


def forward_selection(
    cfg: dict[str, Any],
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    sel_cfg = cfg.get("analysis", {}).get("feature_selection", {})
    model_name = str(sel_cfg.get("model", "ridge"))
    model_params_override = dict(sel_cfg.get("model_params", {}))
    metric = str(sel_cfg.get("metric", "srcc"))
    max_features = min(int(sel_cfg.get("max_features", len(features))), len(features))
    selected: list[str] = []
    remaining = list(features)
    rows = []

    for step in range(1, max_features + 1):
        candidates = []
        for feature in remaining:
            trial = selected + [feature]
            plcc, srcc = evaluate_feature_subset(
                cfg, model_name, model_params_override, X_train, y_train, X_test, y_test, trial
            )
            candidates.append((feature_selection_score(plcc, srcc, metric), feature, plcc, srcc))
        if not candidates:
            break
        _, best_feature, best_plcc, best_srcc = max(candidates, key=lambda item: item[0])
        selected.append(best_feature)
        remaining.remove(best_feature)
        rows.append(
            {
                "direction": "forward",
                "step": step,
                "n_features": len(selected),
                "changed_feature": best_feature,
                "plcc": best_plcc,
                "srcc": best_srcc,
                "features": "|".join(selected),
            }
        )
    return pd.DataFrame(rows)


def backward_elimination(
    cfg: dict[str, Any],
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    sel_cfg = cfg.get("analysis", {}).get("feature_selection", {})
    model_name = str(sel_cfg.get("model", "ridge"))
    model_params_override = dict(sel_cfg.get("model_params", {}))
    metric = str(sel_cfg.get("metric", "srcc"))
    min_features = max(1, int(sel_cfg.get("min_features", 1)))
    selected = list(features)
    rows = []

    initial_plcc, initial_srcc = evaluate_feature_subset(
        cfg, model_name, model_params_override, X_train, y_train, X_test, y_test, selected
    )
    rows.append(
        {
            "direction": "backward",
            "step": 0,
            "n_features": len(selected),
            "changed_feature": "",
            "plcc": initial_plcc,
            "srcc": initial_srcc,
            "features": "|".join(selected),
        }
    )

    step = 0
    while len(selected) > min_features:
        step += 1
        candidates = []
        for feature in selected:
            trial = [item for item in selected if item != feature]
            plcc, srcc = evaluate_feature_subset(
                cfg, model_name, model_params_override, X_train, y_train, X_test, y_test, trial
            )
            candidates.append((feature_selection_score(plcc, srcc, metric), feature, plcc, srcc, trial))
        _, removed_feature, best_plcc, best_srcc, selected = max(candidates, key=lambda item: item[0])
        rows.append(
            {
                "direction": "backward",
                "step": step,
                "n_features": len(selected),
                "changed_feature": removed_feature,
                "plcc": best_plcc,
                "srcc": best_srcc,
                "features": "|".join(selected),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_selection(selection_df: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> Path | None:
    if selection_df.empty:
        return None

    metric = str(cfg.get("analysis", {}).get("feature_selection", {}).get("metric", "srcc"))
    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg.get("plot", {}).get("feature_selection_figsize", [8, 5])))
        for direction, group in selection_df.groupby("direction"):
            group = group.sort_values("n_features")
            ax.plot(group["n_features"], group[metric], marker="o", linewidth=1.8, label=direction)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(metric.upper())
        ax.set_title("Feature Selection Performance")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

    out_path = out_dir / "feature_selection_performance.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def save_feature_selection_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str | None]:
    selection_dir = ensure_dir(out_dir / "feature_selection")
    sel_cfg = cfg.get("analysis", {}).get("feature_selection", {})
    features = resolve_feature_specs(sel_cfg.get("features"), X_train.columns)
    if not features:
        raise ValueError("Feature selection has no matching features")
    max_features = sel_cfg.get("max_features")
    if max_features is not None:
        features = features[: max(1, int(max_features))]
    directions = set(_config_list(sel_cfg.get("directions", ["forward"])))

    frames = []
    if "forward" in directions:
        frames.append(forward_selection(cfg, features, X_train, y_train, X_test, y_test))
    if "backward" in directions:
        frames.append(backward_elimination(cfg, features, X_train, y_train, X_test, y_test))

    selection_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    selection_csv = selection_dir / "feature_selection.csv"
    selection_df.to_csv(selection_csv, index=False)
    plot_path = plot_feature_selection(selection_df, selection_dir, cfg)
    return {
        "feature_selection_csv": str(selection_csv),
        "feature_selection_plot": str(plot_path) if plot_path else None,
    }


def run_experiment(cfg: dict[str, Any], make_plots: bool = True) -> dict[str, Any]:
    np.random.seed(cfg["seed"])

    dataset = build_dataset(cfg)
    X_train, X_test, y_train, y_test = split_dataset(dataset, cfg)

    try:
        run_name = f"{cfg['experiment_name']}@pca{cfg['features']['pca_n']}"
    except KeyError:
        run_name = f"{cfg['experiment_name']}"
    out_dir = ensure_dir(Path(cfg["paths"]["plots_root"]) / run_name)
    output_dirs = regressor_output_dirs(out_dir)

    if cfg.get("save_dataset_snapshot", False):
        relativize_path_columns(dataset).to_csv(output_dirs["metadata"] / "dataset_snapshot.csv", index=False)

    results = []
    importance_paths: dict[str, str | None] = {}
    shap_paths: dict[str, str | None] = {}
    all_plcc: list[float] = []
    all_srcc: list[float] = []
    profile_regressors = is_regressor_profiling_enabled(cfg)
    profile_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}
    prediction_names = dataset.loc[y_test.index, cfg["dataset"]["name_column"]].reset_index(drop=True)

    for model_name, model in init_models(cfg):
        if profile_regressors:
            train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_runtime_sec = time.perf_counter() - train_start

            predict_start = time.perf_counter()
            pred = model.predict(X_test)
            predict_runtime_sec = time.perf_counter() - predict_start
            profile_rows.append(
                build_regressor_profile_row(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    train_runtime_sec=train_runtime_sec,
                    predict_runtime_sec=predict_runtime_sec,
                )
            )
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        plcc, srcc = safe_corr(y_test, pred)
        all_plcc.append(plcc)
        all_srcc.append(srcc)
        predictions_by_model[model_name] = pd.DataFrame(
            {
                "name": prediction_names,
                "mos": y_test.reset_index(drop=True),
                "prediction": np.asarray(pred, dtype=float).reshape(-1),
            }
        )

        results.append({"model": model_name, "plcc": plcc, "srcc": srcc, "source": "regressor"})
        if make_plots and plot_enabled(cfg, "importance"):
            imp_path = plot_importance(model_name, model, X_test, y_test, output_dirs["importances"], cfg)
            importance_paths[model_name] = str(imp_path) if imp_path else None
        if make_plots and plot_enabled(cfg, "shap"):
            shap_path = plot_shap_importance(model_name, model, X_test, output_dirs["shap"], cfg)
            shap_paths[model_name] = str(shap_path) if shap_path else None

    results.extend(compute_metric_comparisons(cfg, dataset, y_test))

    if cfg["save_mean_correlations"]:
        results.append(
            {
                "model": "mean",
                "plcc": float(np.nanmean(all_plcc)),
                "srcc": float(np.nanmean(all_srcc)),
                "source": "summary",
            }
        )
    if cfg["save_best_correlations"]:
        results.append(
            {
                "model": "best",
                "plcc": float(np.max(all_plcc)),
                "srcc": float(np.max(all_srcc)),
                "source": "summary",
            }
        )

    results_df = pd.DataFrame(results).sort_values("srcc", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dirs["correlations"] / "correlations.csv", index=False)
    for model_name, predictions in predictions_by_model.items():
        predictions.to_csv(output_dirs["predictions"] / f"predictions_{model_name}.csv", index=False)

    regressor_profile_path = None
    regressor_total_profile_path = None
    feature_profile_summary_path = None
    if profile_rows:
        regressor_profile_path = resolve_regressor_profile_path(cfg, out_dir, run_name)
        ensure_dir(regressor_profile_path.parent)
        regressor_profile = pd.DataFrame(profile_rows)
        regressor_profile.to_csv(regressor_profile_path, index=False)

        feature_profile_summary = load_feature_profile_summary(cfg, X_train.columns)
        if not feature_profile_summary.empty:
            feature_profile_summary_path = output_dirs["profiling"] / "regressor_feature_profile_summary.csv"
            feature_profile_summary.to_csv(feature_profile_summary_path, index=False)

            total_profile = build_regressor_total_profile(regressor_profile, feature_profile_summary)
            if not total_profile.empty:
                regressor_total_profile_path = resolve_regressor_total_profile_path(cfg, out_dir, run_name)
                ensure_dir(regressor_total_profile_path.parent)
                total_profile.to_csv(regressor_total_profile_path, index=False)

    feature_correlations = compute_feature_correlations(X_test, y_test, cfg)
    feature_correlations.to_csv(output_dirs["feature_analysis"] / "feature_correlations.csv", index=False)
    X_all = pd.concat([X_train, X_test], axis=0).sort_index()
    feature_cross_correlations = compute_feature_cross_correlations(X_all, cfg)
    feature_cross_correlations.to_csv(output_dirs["feature_analysis"] / "feature_cross_correlations.csv")

    analysis_paths: dict[str, str | None] = {}
    name_col = cfg["dataset"]["name_column"]
    names_all = dataset.loc[X_all.index, name_col].reset_index(drop=True)
    if analysis_enabled(cfg, "outliers"):
        analysis_paths.update(save_outlier_analysis(X_all, names_all, predictions_by_model, out_dir, cfg))
    if analysis_enabled(cfg, "feature_metrics"):
        analysis_paths.update(save_feature_analysis_metrics(X_train, y_train, out_dir, cfg))
    if analysis_enabled(cfg, "feature_selection"):
        analysis_paths.update(save_feature_selection_analysis(X_train, y_train, X_test, y_test, out_dir, cfg))

    combined_importance_path = None
    combined_shap_importance_path = None
    correlations_path = None
    correlations_without_metrics_path = None
    feature_correlations_path = None
    feature_cross_correlations_path = None
    prediction_scatter_path = None
    if make_plots:
        if plot_enabled(cfg, "all_importances"):
            combined_importance_path = plot_all_importances(importance_paths, output_dirs["importances"], cfg)
        if plot_enabled(cfg, "all_shap_importances"):
            combined_shap_importance_path = plot_all_shap_importances(shap_paths, output_dirs["shap"], cfg)
        if plot_enabled(cfg, "correlations"):
            correlations_path = plot_correlations(results_df, output_dirs["correlations"], cfg)
        if plot_enabled(cfg, "feature_correlations"):
            feature_correlations_path = plot_feature_correlations(
                feature_correlations,
                results_df,
                output_dirs["feature_analysis"],
                cfg,
            )
        if plot_enabled(cfg, "feature_cross_correlation_matrix"):
            feature_cross_correlations_path = plot_feature_cross_correlation_matrix(
                feature_cross_correlations,
                output_dirs["feature_analysis"],
                cfg,
            )
        if plot_enabled(cfg, "prediction_scatter"):
            prediction_scatter_path = plot_prediction_scatter(predictions_by_model, output_dirs["predictions"], cfg)
        if "source" in results_df.columns:
            without_metrics = results_df[results_df["source"] != "metric"].copy()
        else:
            without_metrics = results_df.copy()
        if plot_enabled(cfg, "correlations_without_metrics") and not without_metrics.empty:
            correlations_without_metrics_path = plot_correlations(
                without_metrics,
                output_dirs["correlations"],
                cfg,
                filename="correlations_without_metrics.png",
                title="Regressor Correlation Scores",
            )

    with open(output_dirs["metadata"] / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    return {
        "dataset": dataset,
        "results": results_df,
        "output_dir": out_dir,
        "output_dirs": {name: str(path) for name, path in output_dirs.items()},
        "importance_paths": importance_paths,
        "shap_paths": shap_paths,
        "analysis_paths": analysis_paths,
        "all_importances_path": str(combined_importance_path) if combined_importance_path else None,
        "all_shap_importances_path": (
            str(combined_shap_importance_path) if combined_shap_importance_path else None
        ),
        "correlations_path": str(correlations_path) if correlations_path else None,
        "correlations_without_metrics_path": (
            str(correlations_without_metrics_path) if correlations_without_metrics_path else None
        ),
        "feature_correlations_path": str(feature_correlations_path) if feature_correlations_path else None,
        "feature_cross_correlations_path": (
            str(feature_cross_correlations_path) if feature_cross_correlations_path else None
        ),
        "prediction_scatter_path": str(prediction_scatter_path) if prediction_scatter_path else None,
        "regressor_profile_path": str(regressor_profile_path) if regressor_profile_path else None,
        "regressor_total_profile_path": (
            str(regressor_total_profile_path) if regressor_total_profile_path else None
        ),
        "feature_profile_summary_path": str(feature_profile_summary_path) if feature_profile_summary_path else None,
    }


def extract_regressor_config(cfg: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    section = cfg.get("regressors")
    if not isinstance(section, dict):
        return cfg

    controls = {
        "enabled",
        "make_plots",
        "no_plots",
        "config_path",
        "config",
        "overrides",
    }
    regressor_keys = {
        "seed",
        "experiment_name",
        "test_size",
        "scale_features",
        "permutation_repeats",
        "paths",
        "score_preparation",
        "dataset",
        "features",
        "models",
    }

    if "config_path" in section:
        nested_path = Path(section["config_path"]).expanduser()
        if not nested_path.is_absolute() and base_dir is not None:
            nested_path = base_dir / nested_path
        result = load_config(nested_path)
    elif isinstance(section.get("config"), dict):
        result = deepcopy(section["config"])
    elif any(key in section for key in regressor_keys):
        result = {key: deepcopy(value) for key, value in section.items() if key not in controls}
    else:
        return cfg

    if isinstance(section.get("config"), dict) and "config_path" in section:
        result = deep_update(result, section["config"])
    if isinstance(section.get("overrides"), dict):
        result = deep_update(result, section["overrides"])

    inferred_categories = infer_feature_categories_from_pipeline_config(cfg)
    if inferred_categories:
        result = deep_update({"feature_categories": inferred_categories}, result)

    return result


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return extract_regressor_config(json.load(handle), path.parent)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured QualiSR-Lab regressor experiments.")
    parser.add_argument("--config", default="configs/default.json", help="Path to experiment JSON config.")
    parser.add_argument("--experiment-name", default=None, help="Override config experiment_name.")
    parser.add_argument("--plots-root", default=None, help="Override config paths.plots_root.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--save-svg", action="store_true", help="Also save generated plots in SVG format.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Measure regressor train/predict runtime and save regressor_profile.csv.",
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        help=(
            "Output CSV path for regressor runtime/FLOPs profile. "
            "Implies --profile. Default: <run_output>/profiling/regressor_profile.csv."
        ),
    )
    parser.add_argument(
        "--profile-total-output",
        default=None,
        help=(
            "Output CSV path for feature+regressor runtime/FLOPs totals when feature profile data exists. "
            "Default: <run_output>/profiling/regressor_total_profile.csv."
        ),
    )
    parser.add_argument(
        "--feature-profile-files",
        nargs="+",
        default=None,
        help=(
            "Existing feature profile CSV files to aggregate into regressor totals. "
            "Can also be configured as profiling.feature_profile_files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))

    overrides: dict[str, Any] = {}
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.plots_root is not None:
        overrides.setdefault("paths", {})["plots_root"] = args.plots_root
    if args.save_svg:
        overrides.setdefault("plot", {})["save_svg"] = True
    if (
        args.profile
        or args.profile_output is not None
        or args.profile_total_output is not None
        or args.feature_profile_files is not None
    ):
        overrides.setdefault("profiling", {})["regressors"] = True
    if args.profile_output is not None:
        overrides.setdefault("profiling", {})["regressor_output"] = args.profile_output
    if args.profile_total_output is not None:
        overrides.setdefault("profiling", {})["regressor_total_output"] = args.profile_total_output
    if args.feature_profile_files is not None:
        overrides.setdefault("profiling", {})["feature_profile_files"] = args.feature_profile_files
    if overrides:
        cfg = deep_update(cfg, overrides)

    result = run_experiment(cfg, make_plots=not args.no_plots)
    print(f"Saved results to {result['output_dir']}")
    print(result["results"].to_string(index=False))


if __name__ == "__main__":
    main()
