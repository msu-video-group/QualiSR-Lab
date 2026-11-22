"""Training-partition feature preparation for the paper experiments."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


class FoldTransformer(BaseEstimator):
    """Fit paired PCA, imputation, and scaling on one training partition.

    Each embedding family uses one PCA basis fitted to stacked SR and reference
    training embeddings, as in the paired mode of :mod:`qualisr.pca`. ``project``
    exposes the unscaled coordinates and signed differences; ``transform`` also
    applies the training-fitted scalar imputer and feature scaler. The caller is
    responsible for passing only permitted training rows to ``fit``.
    """

    def __init__(
        self,
        scalar_columns: list[str],
        embedding_blocks: dict[str, tuple[list[str], list[str]]],
        dimensions: dict[str, int],
        seed: int = 42,
        scaler: str = "minmax",
    ) -> None:
        self.scalar_columns = scalar_columns
        self.embedding_blocks = embedding_blocks
        self.dimensions = dimensions
        self.seed = seed
        self.scaler = scaler

    def _validate_configuration(self) -> None:
        if self.scaler not in {"minmax", "standard"}:
            raise ValueError(f"Unknown feature scaler: {self.scaler}")
        if set(self.embedding_blocks) != set(self.dimensions):
            raise ValueError("PCA dimensions must name exactly the embedding families")
        if not self.scalar_columns and not self.embedding_blocks:
            raise ValueError("At least one scalar or embedding feature is required")
        if len(self.scalar_columns) != len(set(self.scalar_columns)):
            raise ValueError("Scalar feature names must be unique")
        for family, (sr_columns, reference_columns) in self.embedding_blocks.items():
            if not sr_columns or len(sr_columns) != len(reference_columns):
                raise ValueError(f"Embedding family '{family}' requires equally sized nonempty paired blocks")
            columns = sr_columns + reference_columns
            if len(columns) != len(set(columns)):
                raise ValueError(f"Embedding family '{family}' has duplicated or overlapping paired columns")
            dimension = self.dimensions[family]
            if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
                raise ValueError(f"PCA dimension for '{family}' must be a positive integer")

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        required = ["sample_id", *self.scalar_columns]
        for sr_columns, reference_columns in self.embedding_blocks.values():
            required.extend(sr_columns)
            required.extend(reference_columns)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"Feature frame is missing required columns: {missing}")
        if frame.columns.duplicated().any():
            raise ValueError("Feature frame has duplicate column names")
        if frame["sample_id"].isna().any():
            raise ValueError("Feature frame contains missing sample IDs")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Feature frame contains duplicate sample IDs")

    @staticmethod
    def _embedding_values(frame: pd.DataFrame, columns: list[str], family: str) -> np.ndarray:
        values = frame[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Embedding family '{family}' contains nonfinite values")
        return values

    def fit(self, frame: pd.DataFrame) -> FoldTransformer:
        """Fit every learned quantity using exactly the supplied training rows."""
        self._fit_imputed(frame)
        return self

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training rows with one embedding projection."""
        imputed = self._fit_imputed(frame)
        values = self.scaler_.transform(imputed)
        return pd.DataFrame(values, columns=self.feature_names_out_, index=frame.index)

    def _fit_imputed(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate_configuration()
        self._validate_frame(frame)
        if frame.empty:
            raise ValueError("Cannot fit feature preparation on an empty training partition")
        self.fit_sample_ids_ = frame["sample_id"].astype(str).tolist()
        self.pcas_: dict[str, PCA] = {}
        for family, (sr_columns, reference_columns) in self.embedding_blocks.items():
            sr_values = self._embedding_values(frame, sr_columns, family)
            reference_values = self._embedding_values(frame, reference_columns, family)
            paired_values = np.concatenate([sr_values, reference_values], axis=0)
            dimension = int(self.dimensions[family])
            maximum = min(paired_values.shape)
            if dimension > maximum:
                raise ValueError(
                    f"PCA dimension {dimension} for '{family}' exceeds the training-partition maximum {maximum}"
                )
            pca = PCA(n_components=dimension, svd_solver="randomized", random_state=self.seed)
            pca.fit(paired_values)
            self.pcas_[family] = pca

        projected = self.project(frame)
        all_missing = projected.columns[projected.isna().all()].tolist()
        if all_missing:
            raise ValueError(f"Training features contain no finite values: {all_missing}")
        self.feature_names_out_ = projected.columns.tolist()
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(projected)
        scaler = MinMaxScaler() if self.scaler == "minmax" else StandardScaler()
        self.scaler_ = scaler.fit(imputed)
        return imputed

    def project(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return unscaled features, including PCA-coordinate ``SR - reference``."""

        check_is_fitted(self, "pcas_")
        self._validate_frame(frame)
        result = frame[self.scalar_columns].astype(float).replace([np.inf, -np.inf], np.nan).copy()
        for family, (sr_columns, reference_columns) in self.embedding_blocks.items():
            sr_values = self._embedding_values(frame, sr_columns, family)
            reference_values = self._embedding_values(frame, reference_columns, family)
            pca = self.pcas_[family]
            sr = pca.transform(sr_values)
            reference = pca.transform(reference_values)
            for representation, values in [("sr", sr), ("ref", reference), ("diff", sr - reference)]:
                columns = [f"{family}_{representation}_{i:03d}" for i in range(values.shape[1])]
                block = pd.DataFrame(values, columns=columns, index=frame.index)
                result = pd.concat([result, block], axis=1)
        if result.columns.duplicated().any():
            raise ValueError("Scalar and projected embedding feature names overlap")
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prepare held-out or external rows without fitting any new parameters."""

        check_is_fitted(self, ["imputer_", "scaler_", "feature_names_out_"])
        projected = self.project(frame)
        values = self.scaler_.transform(self.imputer_.transform(projected))
        return pd.DataFrame(values, columns=self.feature_names_out_, index=frame.index)
