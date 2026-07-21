from __future__ import annotations

import numpy as np


class RegularizedPCAWhitening:
    """Full-dimensional regularized PCA whitening.

    This follows the audited Christlein author-code logic:
    X_centered @ components.T / sqrt(explained_variance + regularization).
    """

    def __init__(self, regularization: float = 0.001, n_components: int | None = None):
        self.regularization = float(regularization)
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RegularizedPCAWhitening":
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")

        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        x_centered = x - self.mean_

        _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
        explained_variance = (singular_values ** 2) / x.shape[0]

        n_components = self.n_components or x.shape[1]
        n_components = min(n_components, vt.shape[0])

        self.components_ = vt[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None or self.explained_variance_ is None:
            raise RuntimeError("RegularizedPCAWhitening must be fitted before transform().")

        x = np.asarray(x, dtype=np.float64)
        x_centered = x - self.mean_
        x_proj = x_centered @ self.components_.T
        x_white = x_proj / np.sqrt(self.explained_variance_ + self.regularization)
        return x_white.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
