"""Prespecified matched feature pools and bounded joint tuning candidates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.data import FR_METRICS, NR_COLUMNS, STAT_COLUMNS

COMPACT_STAT_COLUMNS = ["max", "median", "std", "area00"]


def _fr_columns(reference: str) -> list[str]:
    return [f"{metric}_{reference}" for metric in FR_METRICS]


def experiment_specs() -> list[dict[str, Any]]:
    """Return the fixed matrix; only the six principal comparisons are tuned."""
    nr = list(NR_COLUMNS)
    fr = _fr_columns("rlfn")
    stats = list(COMPACT_STAT_COLUMNS)
    specs = [
        ("nr", nr, [], True),
        ("nr_fr", nr + fr, [], True),
        ("nr_ref", nr, ["ref"], True),
        ("nr_diff", nr, ["diff"], True),
        ("nr_stats", nr + stats, [], True),
        ("nr_all", nr + fr + stats, ["ref", "diff"], True),
        ("fr_bicubic", _fr_columns("bicubic"), [], False),
        ("fr_rlfn", fr, [], False),
        ("fr_span", _fr_columns("span"), [], False),
        ("nr_fr_bicubic", nr + _fr_columns("bicubic"), [], False),
        ("nr_fr_span", nr + _fr_columns("span"), [], False),
        ("full_with_stats", nr + fr + stats, ["sr", "diff"], False),
        ("full_without_stats", nr + fr, ["sr", "diff"], False),
        ("full_all_statistics", nr + fr + STAT_COLUMNS, ["sr", "diff"], False),
        ("nr_stats_all", nr + STAT_COLUMNS, [], False),
        ("full_without_qalign", [name for name in nr if name != "qalign"] + fr + stats, ["sr", "diff"], False),
    ]
    return [
        {"name": name, "scalars": list(scalars), "representations": list(representations), "tune": tune}
        for name, scalars, representations, tune in specs
    ]


def search_candidates(model: str, has_embeddings: bool) -> list[dict[str, Any]]:
    """Return eight joint settings, including the default backend/PCA-5 control.

    This is a bounded prespecified sensitivity search, not a Cartesian search or
    evidence that the listed settings contain an optimal backend configuration.
    Random seeds and CPU-thread limits are supplied by the experiment runner.
    """
    parameter_sets = {
        "randomforest": [
            {},
            {"max_depth": 10},
            {"max_depth": 20},
            {"max_depth": 10, "min_samples_leaf": 2},
            {"min_samples_leaf": 4, "max_features": 0.7},
            {"max_depth": 20, "min_samples_leaf": 2, "max_features": 0.7},
            {"n_estimators": 300, "max_features": 0.7, "min_samples_leaf": 2},
            {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 4},
        ],
        "xgb": [
            {},
            {"max_depth": 3},
            {"max_depth": 6},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
            {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 500},
            {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 500},
        ],
        "catboost": [
            {},
            {"depth": 4},
            {"depth": 8},
            {"depth": 4, "learning_rate": 0.05, "iterations": 500},
            {"depth": 6, "learning_rate": 0.05, "iterations": 500},
            {"depth": 4, "learning_rate": 0.1, "iterations": 1000},
            {"depth": 8, "learning_rate": 0.05, "iterations": 500},
            {"depth": 6, "learning_rate": 0.1, "iterations": 1000, "l2_leaf_reg": 5},
        ],
    }
    if model not in parameter_sets:
        raise ValueError(f"No prespecified tuning candidates for backend: {model}")
    params = deepcopy(parameter_sets[model])
    dimensions = [(5, 5)] * 8
    if has_embeddings:
        params[1] = {}
        params[2] = {}
        dimensions = [(5, 5), (5, 25), (5, 50), (5, 5), (5, 25), (5, 50), (10, 25), (10, 50)]
    return [
        {"params": setting, "dimensions": {"vgg": vgg, "resnet": resnet}, "stats_columns": None}
        for setting, (vgg, resnet) in zip(params, dimensions, strict=True)
    ]
