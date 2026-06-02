"""Unified JSON-configured pipeline runner for QualiSR-Lab."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


SECTION_ORDER = ("references", "features", "pca", "statistics", "regressors")


def load_pipeline_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def section_enabled(cfg: Mapping[str, Any], default: bool = False) -> bool:
    return bool(cfg.get("enabled", default))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def named_specs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [f"{name}={path}" for name, path in value.items()]

    specs = []
    for item in as_list(value):
        if isinstance(item, Mapping):
            name = item.get("name") or item.get("method") or item.get("ref") or item.get("prefix")
            path = item.get("path") or item.get("dir") or item.get("directory")
            if name is None or path is None:
                raise ValueError(f"Named path entries must define name/path fields: {item}")
            specs.append(f"{name}={path}")
        else:
            specs.append(str(item))
    return specs


def add_value(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def add_csv_value(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        parsed = value
    else:
        parsed = ",".join(str(item) for item in as_list(value))
    if parsed:
        argv.extend([flag, parsed])


def add_values(argv: list[str], flag: str, values: Any) -> None:
    parsed = [str(value) for value in as_list(values)]
    if parsed:
        argv.append(flag)
        argv.extend(parsed)


def add_named_specs(argv: list[str], flag: str, value: Any) -> None:
    specs = named_specs(value)
    if specs:
        argv.append(flag)
        argv.extend(specs)


def add_bool(argv: list[str], flag: str, enabled: Any) -> None:
    if enabled:
        argv.append(flag)


def run_module_main(module_name: str, argv: Sequence[str]) -> None:
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [module_name.rsplit(".", 1)[-1], *argv]
        module.main()
    finally:
        sys.argv = old_argv


def run_references(cfg: Mapping[str, Any]) -> None:
    refs = [str(ref).lower() for ref in as_list(cfg.get("refs", []))]
    if not refs:
        return

    argv: list[str] = []
    add_value(argv, "--lr-dir", cfg.get("lr_dir"))
    add_named_specs(argv, "--sr-dirs", cfg.get("sr_dirs"))
    add_values(argv, "--refs", refs)
    add_named_specs(argv, "--ref-dirs", cfg.get("ref_dirs"))

    for key, flag in [
        ("out_root", "--out-root"),
        ("output_ext", "--output-ext"),
        ("bicubic_suffix", "--bicubic-suffix"),
        ("rlfn_suffix", "--rlfn-suffix"),
        ("span_suffix", "--span-suffix"),
        ("scale", "--scale"),
        ("python_exec", "--python-exec"),
        ("rlfn_script", "--rlfn-script"),
        ("rlfn_ckpt", "--rlfn-ckpt"),
        ("rlfn_cmd_template", "--rlfn-cmd-template"),
        ("span_script", "--span-script"),
        ("span_ckpt", "--span-ckpt"),
        ("span_cmd_template", "--span-cmd-template"),
        ("limit", "--limit"),
        ("log_level", "--log-level"),
    ]:
        add_value(argv, flag, cfg.get(key))

    add_bool(argv, "--overwrite", cfg.get("overwrite"))
    add_bool(argv, "--strict", cfg.get("strict"))
    add_bool(argv, "--no-progress", cfg.get("no_progress"))
    run_module_main("scripts.make_reference", argv)


def feature_group_items(cfg: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    groups = cfg.get("groups", [])
    if isinstance(groups, Mapping):
        return [(str(name), group or {}) for name, group in groups.items()]
    return [
        (str(group.get("name", index)), group)
        for index, group in enumerate(as_list(groups))
        if isinstance(group, Mapping)
    ]


def run_feature_group(common: Mapping[str, Any], name: str, group: Mapping[str, Any]) -> None:
    if not section_enabled(group, default=True):
        return

    merged = deep_update(dict(common), dict(group))
    features = merged.get("features", name)
    if isinstance(features, str):
        features_arg = features
    else:
        features_arg = ",".join(str(feature) for feature in as_list(features))

    output = merged.get("output")
    if output is None:
        raise ValueError(f"Feature group '{name}' must define an output path")

    argv: list[str] = []
    add_named_specs(argv, "--sr-dirs", merged.get("sr_dirs"))
    add_value(argv, "--gt-dir", merged.get("gt_dir"))
    add_value(argv, "--lr-dir", merged.get("lr_dir"))
    add_named_specs(argv, "--ref-dirs", merged.get("ref_dirs"))
    add_value(argv, "--features", features_arg)
    add_csv_value(argv, "--fr-metrics", merged.get("fr_metrics"))
    add_csv_value(argv, "--nr-metrics", merged.get("nr_metrics"))
    add_named_specs(argv, "--timm-encoders", merged.get("timm_encoders"))
    add_value(argv, "--siglip-model", merged.get("siglip_model"))
    add_value(argv, "--siglip-alpha", merged.get("siglip_alpha"))
    add_value(argv, "--noise-components", merged.get("noise_components"))
    add_value(argv, "--noise-seed", merged.get("noise_seed"))
    add_value(argv, "--output", output)
    add_value(argv, "--profile-output", merged.get("profile_output"))
    add_value(argv, "--device", merged.get("device"))
    add_value(argv, "--log-level", merged.get("log_level"))
    add_bool(argv, "--profile", merged.get("profile"))
    add_bool(argv, "--profile-flops", merged.get("profile_flops"))
    add_bool(argv, "--timm-no-pretrained", merged.get("timm_no_pretrained"))
    add_bool(argv, "--strict", merged.get("strict"))
    run_module_main("scripts.get_image_features", argv)


def run_features(cfg: Mapping[str, Any]) -> None:
    common = cfg.get("common", {})
    if not isinstance(common, Mapping):
        raise ValueError("features.common must be an object")

    for name, group in feature_group_items(cfg):
        run_feature_group(common, name, group)


def pca_run_items(cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = cfg.get("runs")
    if runs is None and "input" in cfg:
        return [cfg]
    if isinstance(runs, Mapping):
        return [run or {} for run in runs.values()]
    return [run for run in as_list(runs) if isinstance(run, Mapping)]


def run_pca(cfg: Mapping[str, Any]) -> None:
    common = cfg.get("common", {})
    if not isinstance(common, Mapping):
        raise ValueError("pca.common must be an object")

    for index, run in enumerate(pca_run_items(cfg)):
        merged = deep_update(dict(common), dict(run))
        if not section_enabled(merged, default=True):
            continue
        if merged.get("input") is None:
            raise ValueError(f"PCA run #{index} must define input")

        argv: list[str] = []
        add_value(argv, "--input", merged.get("input"))
        add_values(argv, "--n-components", merged.get("n_components"))
        add_values(argv, "--blocks", merged.get("blocks"))
        add_value(argv, "--output-dir", merged.get("output_dir"))
        add_value(argv, "--output-template", merged.get("output_template"))
        add_value(argv, "--fit-column", merged.get("fit_column"))
        add_value(argv, "--fit-value", merged.get("fit_value"))
        add_value(argv, "--split-column", merged.get("split_column"))
        add_value(argv, "--train-label", merged.get("train_label"))
        add_value(argv, "--test-label", merged.get("test_label"))
        add_value(argv, "--test-size", merged.get("test_size"))
        add_value(argv, "--split-seed", merged.get("split_seed"))
        add_value(argv, "--group-column", merged.get("group_column"))
        add_value(argv, "--group-fallback-column", merged.get("group_fallback_column"))
        add_value(argv, "--svd-solver", merged.get("svd_solver"))
        add_value(argv, "--log-level", merged.get("log_level"))
        add_bool(argv, "--keep-original-blocks", merged.get("keep_original_blocks"))
        add_bool(argv, "--disable-auto-split", merged.get("disable_auto_split"))
        run_module_main("scripts.apply_pca", argv)


def run_statistics(cfg: Mapping[str, Any]) -> None:
    argv: list[str] = []
    add_named_specs(argv, "--heatmap-dirs", cfg.get("heatmap_dirs"))
    add_value(argv, "--output", cfg.get("output"))
    add_value(argv, "--profile-output", cfg.get("profile_output"))
    add_values(argv, "--percentiles", cfg.get("percentiles"))
    add_values(argv, "--area-thresholds", cfg.get("area_thresholds"))
    add_values(argv, "--extensions", cfg.get("extensions"))
    add_value(argv, "--log-level", cfg.get("log_level"))
    add_bool(argv, "--profile", cfg.get("profile"))
    add_bool(argv, "--recursive", cfg.get("recursive"))
    add_bool(argv, "--strict", cfg.get("strict"))
    add_bool(argv, "--no-progress", cfg.get("no_progress"))
    run_module_main("scripts.compute_statistics", argv)


def collect_feature_categories(features_cfg: Mapping[str, Any] | None) -> dict[str, list[str]]:
    if not isinstance(features_cfg, Mapping):
        return {}

    categories: dict[str, list[str]] = {
        "nr_metrics": [],
        "fr_metrics": [],
        "timm_prefixes": [],
    }

    sources: list[Mapping[str, Any]] = []
    common = features_cfg.get("common", {})
    if isinstance(common, Mapping):
        sources.append(common)

    groups = features_cfg.get("groups", {})
    if isinstance(groups, Mapping):
        sources.extend(group for group in groups.values() if isinstance(group, Mapping))
    else:
        sources.extend(group for group in as_list(groups) if isinstance(group, Mapping))

    for source in sources:
        categories["nr_metrics"].extend(_config_list_for_categories(source.get("nr_metrics")))
        categories["fr_metrics"].extend(_config_list_for_categories(source.get("fr_metrics")))
        timm_encoders = source.get("timm_encoders")
        if isinstance(timm_encoders, Mapping):
            categories["timm_prefixes"].extend(str(name) for name in timm_encoders)
        else:
            for spec in _config_list_for_categories(timm_encoders):
                categories["timm_prefixes"].append(spec.split("=", 1)[0].strip())

    return {
        key: sorted(set(value), key=str.lower)
        for key, value in categories.items()
        if value
    }


def _config_list_for_categories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    return [str(item) for item in as_list(value) if str(item).strip()]


def run_regressors_section(
    cfg: Mapping[str, Any],
    base_dir: Path,
    args: argparse.Namespace,
    features_cfg: Mapping[str, Any] | None = None,
) -> None:
    from qualisr_lab.regressors import extract_regressor_config, run_experiment

    regressor_cfg = extract_regressor_config({"regressors": dict(cfg)}, base_dir)
    inferred_categories = collect_feature_categories(features_cfg)
    if inferred_categories:
        regressor_cfg = deep_update(
            {"feature_categories": inferred_categories},
            regressor_cfg,
        )
    overrides: dict[str, Any] = {}

    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.plots_root is not None:
        overrides.setdefault("paths", {})["plots_root"] = args.plots_root
    if args.save_svg:
        overrides.setdefault("plot", {})["save_svg"] = True
    if overrides:
        regressor_cfg = deep_update(regressor_cfg, overrides)

    make_plots = bool(cfg.get("make_plots", True)) and not args.no_plots
    result = run_experiment(regressor_cfg, make_plots=make_plots)
    print(f"Saved regressor results to {result['output_dir']}")
    print(result["results"].to_string(index=False))


def run_pipeline(cfg: dict[str, Any], config_path: Path, args: argparse.Namespace) -> None:
    selected = set(args.only_section or SECTION_ORDER)
    skipped = set(args.skip_section or [])

    for section_name in SECTION_ORDER:
        if section_name not in selected or section_name in skipped:
            continue
        section_cfg = cfg.get(section_name, {})
        if not isinstance(section_cfg, Mapping) or not section_enabled(section_cfg):
            continue

        if section_name == "references":
            run_references(section_cfg)
        elif section_name == "features":
            run_features(section_cfg)
        elif section_name == "pca":
            run_pca(section_cfg)
        elif section_name == "statistics":
            run_statistics(section_cfg)
        elif section_name == "regressors":
            features_cfg = cfg.get("features", {})
            run_regressors_section(section_cfg, config_path.parent, args, features_cfg=features_cfg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unified QualiSR-Lab pipeline config.")
    parser.add_argument("--config", default="configs/pipeline.json", help="Unified pipeline JSON config.")
    parser.add_argument("--only-section", nargs="+", choices=SECTION_ORDER, default=None)
    parser.add_argument("--skip-section", nargs="+", choices=SECTION_ORDER, default=None)
    parser.add_argument("--experiment-name", default=None, help="Override nested regressor experiment_name.")
    parser.add_argument("--plots-root", default=None, help="Override nested regressor paths.plots_root.")
    parser.add_argument("--no-plots", action="store_true", help="Skip regressor plot generation.")
    parser.add_argument("--save-svg", action="store_true", help="Also save regressor plots in SVG format.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = load_pipeline_config(config_path)
    run_pipeline(cfg, config_path, args)


if __name__ == "__main__":
    main()
