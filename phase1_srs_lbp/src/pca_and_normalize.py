from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.decomposition import PCA


PowerMode = Literal["signed_sqrt", "sqrt_nonneg_clip", "none"]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def power_transform(x: np.ndarray, mode: PowerMode) -> np.ndarray:
    """
    Apply a Hellinger-like / power normalization.

    Notes
    -----
    Classic Hellinger for histograms assumes non-negative vectors and uses sqrt(x).
    After PCA, components can be negative. A common robust choice is signed sqrt:
        sign(x) * sqrt(|x|)
    We expose both to allow verification.
    """
    if mode == "none":
        return x
    if mode == "sqrt_nonneg_clip":
        return np.sqrt(np.clip(x, 0.0, None))
    if mode == "signed_sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    raise ValueError(f"Unknown power mode: {mode}")


@dataclass
class PCANormalizer:
    """
    PCA projection + power transform + L2 normalization.

    Pipeline from the paper:
    - project descriptor onto first N PCs
    - apply Hellinger kernel
    - L2 normalize
    """
    n_components: int = 200
    whiten: bool = False
    power_mode: PowerMode = "signed_sqrt"
    random_state: int = 0

    pca_: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "PCANormalizer":
        if X.ndim != 2:
            raise ValueError("X must be 2D: (n_samples, n_features).")
        pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        pca.fit(X)
        self.pca_ = pca
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCANormalizer must be fit() before transform().")
        Z = self.pca_.transform(X).astype(np.float32, copy=False)
        Z = power_transform(Z, self.power_mode).astype(np.float32, copy=False)
        Z = l2_normalize(Z).astype(np.float32, copy=False)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
