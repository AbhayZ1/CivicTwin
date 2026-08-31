from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

EPS = 1e-8


@dataclass
class FeatureScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: Optional[Sequence[str]] = None

    @property
    def n_features(self) -> int:
        return int(self.mean_.shape[0])

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "FeatureScaler":
        array = np.asarray(data, dtype=np.float64)
        if array.ndim < 2:
            raise ValueError("data must have at least 2 dimensions (..., F)")
        flat = array.reshape(-1, array.shape[-1])
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale = np.where(scale < EPS, 1.0, scale)
        return cls(
            mean_=mean,
            scale_=scale,
            feature_names=list(feature_names) if feature_names else None,
        )

    @classmethod
    def fit_windows(
        cls,
        windows: Iterable[np.ndarray],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "FeatureScaler":
        stacked = np.concatenate(
            [
                np.asarray(w, dtype=np.float64).reshape(-1, np.asarray(w).shape[-1])
                for w in windows
            ],
            axis=0,
        )
        return cls.fit(stacked, feature_names=feature_names)

    def transform(self, data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float64)
        self._check(array)
        return (array - self.mean_) / self.scale_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float64)
        self._check(array)
        return array * self.scale_ + self.mean_

    def _check(self, array: np.ndarray) -> None:
        if array.shape[-1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features on the last axis, got {array.shape[-1]}"
            )


def zscore(data: np.ndarray) -> np.ndarray:
    return FeatureScaler.fit(data).transform(data)


__all__ = ["FeatureScaler", "zscore"]
