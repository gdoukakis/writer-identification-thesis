from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances_argmin


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply row-wise L2 normalization."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def signed_square_root(x: np.ndarray) -> np.ndarray:
    """Apply signed square-root normalization."""
    return np.sign(x) * np.sqrt(np.abs(x))


def hard_vlad_encode(
    descriptors: np.ndarray,
    centers: np.ndarray,
    apply_ssr: bool = True,
    apply_l2: bool = True,
) -> np.ndarray:
    """Compute hard-assignment VLAD for one page/image.

    This intentionally does not apply intra-normalization, matching the audited
    Christlein vlad_enc_ssr configuration.
    """
    descriptors = np.asarray(descriptors, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)

    if descriptors.ndim != 2:
        raise ValueError(f"Expected descriptors with shape (N, D), got {descriptors.shape}")
    if centers.ndim != 2:
        raise ValueError(f"Expected centers with shape (K, D), got {centers.shape}")
    if descriptors.shape[1] != centers.shape[1]:
        raise ValueError(f"Descriptor dim {descriptors.shape[1]} != center dim {centers.shape[1]}")

    k, d = centers.shape
    assignments = pairwise_distances_argmin(descriptors, centers, metric="euclidean")

    vlad = np.zeros((k, d), dtype=np.float32)
    for cluster_id in range(k):
        mask = assignments == cluster_id
        if np.any(mask):
            residuals = descriptors[mask] - centers[cluster_id]
            vlad[cluster_id] = residuals.sum(axis=0)

    vlad_flat = vlad.reshape(1, -1)

    if apply_ssr:
        vlad_flat = signed_square_root(vlad_flat)

    if apply_l2:
        vlad_flat = l2_normalize(vlad_flat)

    return vlad_flat.astype(np.float32)