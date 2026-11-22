"""Subset selection confined to supplied inner training/validation folds."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from qualisr.regressors import validation_correlations

InnerFold = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]


def select_ridge_features(
    inner_folds: list[InnerFold],
    candidates: list[str],
    alpha: float = 1.0,
) -> tuple[list[str], pd.DataFrame]:
    """Run the original nested Ridge selection with its unchanged public API."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    return select_features(inner_folds, candidates, lambda: Ridge(alpha=alpha))


def select_features(
    inner_folds: list[InnerFold],
    candidates: list[str],
    estimator_factory: Callable[[], Any],
    on_trial: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Choose a subset from complete greedy forward and backward inner-CV paths.

    Each fold contains training X/y, validation X/y, and validation metadata.
    Its feature matrices must already be transformed using that fold's training
    rows only. The function has no outer-evaluation input. The objective is the
    arithmetic mean of the per-fold dataset-macro SRCC. A trial is ineligible
    when any inner fold has an undefined macro SRCC; folds are never dropped.
    Ties prefer fewer features, forward selection, then lexical feature order.
    """
    if not inner_folds:
        raise ValueError("Feature selection requires at least one inner fold")
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Feature candidates must be nonempty and unique")

    for train_x, train_y, valid_x, valid_y, metadata in inner_folds:
        if not train_x.columns.is_unique or not valid_x.columns.is_unique:
            raise ValueError("Inner-fold feature columns must be unique")
        missing = sorted(set(candidates) - (set(train_x.columns) & set(valid_x.columns)))
        if missing:
            raise ValueError(f"Inner fold is missing candidate columns: {missing}")
        if len(train_x) != len(train_y) or len(valid_x) != len(valid_y) or len(valid_x) != len(metadata):
            raise ValueError("Inner-fold feature, target, and metadata lengths must agree")
        if train_x.empty or valid_x.empty:
            raise ValueError("Inner training and validation folds must be nonempty")

    cache: dict[tuple[str, ...], dict[str, object]] = {}

    def evaluate(features: list[str]) -> dict[str, object]:
        key = tuple(sorted(features))
        if key not in cache:
            plccs, srccs = [], []
            for train_x, train_y, valid_x, valid_y, metadata in inner_folds:
                model = estimator_factory()
                model.fit(train_x.loc[:, key], train_y)
                prediction = model.predict(valid_x.loc[:, key])
                plcc, srcc, _, _ = validation_correlations(valid_y, prediction, metadata)
                plccs.append(plcc)
                srccs.append(srcc)
            cache[key] = {
                "n_features": len(key),
                "features": "|".join(key),
                "mean_plcc": float(np.mean(plccs)),
                "mean_srcc": float(np.mean(srccs)),
                "fold_plcc": json.dumps(plccs),
                "fold_srcc": json.dumps(srccs),
            }
            if on_trial is not None:
                on_trial(dict(cache[key]))
        return dict(cache[key])

    def best_trial(trials: list[list[str]]) -> tuple[list[str], dict[str, object]]:
        evaluated = [(sorted(features), evaluate(features)) for features in trials]
        eligible = [(features, row) for features, row in evaluated if np.isfinite(row["mean_srcc"])]
        if not eligible:
            raise ValueError("No feature subset has a finite macro SRCC in every inner fold")
        return min(eligible, key=lambda trial: (-trial[1]["mean_srcc"], tuple(trial[0])))

    history: list[dict[str, object]] = []
    selected: list[str] = []
    remaining = sorted(candidates)
    while remaining:
        selected, row = best_trial([selected + [feature] for feature in remaining])
        remaining = [feature for feature in remaining if feature not in selected]
        history.append({"direction": "forward", **row})

    selected = sorted(candidates)
    initial = evaluate(selected)
    if not np.isfinite(initial["mean_srcc"]):
        raise ValueError("The complete candidate pool has undefined macro SRCC in an inner fold")
    history.append({"direction": "backward", **initial})
    while len(selected) > 1:
        selected, row = best_trial([[feature for feature in selected if feature != removed] for removed in selected])
        history.append({"direction": "backward", **row})

    best = min(
        range(len(history)),
        key=lambda index: (
            -history[index]["mean_srcc"],
            history[index]["n_features"],
            history[index]["direction"] != "forward",
            history[index]["features"],
        ),
    )
    result = pd.DataFrame(history)
    result["selected"] = np.arange(len(result)) == best
    return str(history[best]["features"]).split("|"), result
